# 网络排障角色扮演 A：复用的连接偶尔被重置

## 题目

面试官扮演值班同事，手里有日志、监控和机器权限；你来主导排查：提问、要数据、让他跑命令、给出判断。每一步的题目里写着上一步面试官透露的信息，可以一步一步往下练；场景、数字和命令输出都是我们编的，输出格式按真实工具核对过，数值不是实测。

### 开场：缩小范围

**Q1.** 同事说：“订单服务（order service）跑在 AWS 的一个私有子网（private subnet）里，用 Python 调合作方的支付接口 `https://api.partner.example`。大约 0.3% 的调用失败，报下面这个错，重试一次基本就好。合作方说他们那边一切正常。”你先问什么、要什么数据？

```text
requests.exceptions.ConnectionError: ('Connection aborted.', ConnectionResetError(104, 'Connection reset by peer'))
```

**Q2.** 你问到的情况：(1) 大约三周前开始，那时夜间批处理任务搬去了另一个服务，前后没有发版；(2) 所有机器都有，失败集中在 UTC 01:00–06:00，正是流量最低的时候；(3) 失败的请求都是发出后 1 毫秒之内就报错，而且都用的是从连接池里取出的旧连接，新建的连接从来没失败过；(4) 合作方查不到失败请求的任何记录，他们负载均衡器的空闲超时是 1 小时。同事按你的要求，从客户端日志里统计了一周的请求，按“这个连接上一次使用到这次使用隔了多久”（下面叫空闲时长）分组；区间含左端、不含右端。他还查了 DNS。

| 连接空闲时长 | 请求数 | 失败数 |
|---|---:|---:|
| 0–60 秒 | 1,180,000 | 0 |
| 60–300 秒 | 76,500 | 0 |
| 300–350 秒 | 9,300 | 0 |
| 350–600 秒 | 2,940 | 2,940 |
| 600 秒以上 | 860 | 860 |

```text
$ dig +noall +answer api.partner.example
api.partner.example.	60	IN	A	203.0.113.50
api.partner.example.	60	IN	A	203.0.113.51
```

你从这些数据里读出什么？有哪些假设，哪个最符合，下一步做什么？

### 抓包：谁发的 RST

**Q3.** 同事在一台机器（10.0.12.34）上抓了一夜的包：`sudo tcpdump -nn -i ens5 -w partner.pcap 'host 203.0.113.50 and tcp port 443'`，然后挑出一条失败的连接（本地端口 48316）给你看。开头是三次握手：

```text
$ TZ=UTC tcpdump -nn -r partner.pcap 'tcp port 48316'
02:07:14.201455 IP 10.0.12.34.48316 > 203.0.113.50.443: Flags [S], seq 1946283124, win 64240, options [mss 1460,sackOK,TS val 1193046212 ecr 0,nop,wscale 7], length 0
02:07:14.239611 IP 203.0.113.50.443 > 10.0.12.34.48316: Flags [S.], seq 3772190654, ack 1946283125, win 65160, options [mss 1460,sackOK,TS val 3320158872 ecr 1193046212,nop,wscale 7], length 0
02:07:14.239652 IP 10.0.12.34.48316 > 203.0.113.50.443: Flags [.], ack 1, win 502, options [nop,nop,TS val 1193046250 ecr 3320158872], length 0
```

中间 TLS 握手的几行省略了，下面是这条连接剩下的全部内容：

```text
02:07:14.363504 IP 10.0.12.34.48316 > 203.0.113.50.443: Flags [P.], seq 598:1121, ack 4467, win 501, options [nop,nop,TS val 1193046374 ecr 3320158958], length 523
02:07:14.401871 IP 203.0.113.50.443 > 10.0.12.34.48316: Flags [P.], seq 4467:5162, ack 1121, win 501, options [nop,nop,TS val 3320159034 ecr 1193046374], length 695
02:07:14.401899 IP 10.0.12.34.48316 > 203.0.113.50.443: Flags [.], ack 5162, win 497, options [nop,nop,TS val 1193046413 ecr 3320159034], length 0
02:14:07.482913 IP 10.0.12.34.48316 > 203.0.113.50.443: Flags [P.], seq 1121:1533, ack 5162, win 497, options [nop,nop,TS val 1193459494 ecr 3320159034], length 412
02:14:07.483329 IP 203.0.113.50.443 > 10.0.12.34.48316: Flags [R], seq 3772195816, win 0, length 0
```

逐行读一下：RST 是谁发的，大概从哪里来，为什么是这个时候？

**Q4.** 你问这台机器的流量怎么出去。同事说：这个私有子网的默认路由（0.0.0.0/0）指向一个 NAT 网关（NAT gateway），去合作方的流量都经过它。你怎么确认自己的判断，而不是只凭推理？

### 解释与修复

**Q5.** 你的确认结果：同一台机器上，连接空闲 340 秒后再发请求，20 次都成功；空闲 360 秒，20 次都被重置。从一台有公网 IP、不经过 NAT 网关的机器做同样的测试，空闲 900 秒也正常。NAT 网关的 CloudWatch 指标 `IdleTimeoutCount` 一直大于 0，每天夜里有高峰。同事说：“奇怪，我们明明开了 TCP keepalive。”并给你看：

```text
$ sysctl net.ipv4.tcp_keepalive_time net.ipv4.tcp_keepalive_intvl net.ipv4.tcp_keepalive_probes
net.ipv4.tcp_keepalive_time = 7200
net.ipv4.tcp_keepalive_intvl = 75
net.ipv4.tcp_keepalive_probes = 9
$ ss -tno state established dst 203.0.113.50
Recv-Q Send-Q Local Address:Port  Peer Address:Port Process
0      0         10.0.12.34:48402 203.0.113.50:443
0      0         10.0.12.34:48517 203.0.113.50:443
```

为什么 keepalive 没起作用？客户端库从连接池取连接时会先检查连接还在不在，为什么也没拦住？

**Q6.** 你问客户端的写法。同事说：用的是 Python `requests`（底层是 urllib3 2.x），一个全局的 `Session`，默认的 `HTTPAdapter`，没有设置任何套接字选项；失败的调用里既有查询订单状态的 GET，也有创建支付的 POST。今晚就要止血，你怎么改？怎么验证改好了？

**Q7.** 改动上线后，第二天夜里空闲 350 秒以上的那两组请求不再失败，`IdleTimeoutCount` 降到接近 0。面试官问：“长期该怎么防？换到别的云或者别的网络设备上，这个结论还成立吗？如果中间设备超时以后不回 RST，而是悄悄把包丢掉，现象会变成什么样？”

### 复盘

**Q8.** 面试官最后问：“这个问题潜伏了三个星期。怎样才能更早发现，或者在搬走夜间批处理的时候就预见到？”

## 参考解答

<details>
<summary>展开参考解答</summary>

这种轮次的证据很少：流程页只说性能和基础设施岗位的专项轮里有“网络排障角色扮演”，一位性能工程候选人的候选人门户里也列了它；没有人记录过场景、时长和评分方式。下面按排障角色扮演的一般做法来写。面试官大概率看这几件事：会不会主动问对问题；每一步要的数据能不能区分假设；读不读得懂原始输出（tcpdump、ss、sysctl）；先止血再根治；最后能不能复盘。

练习方法：只看题目，先自己说出这一步要问什么、要什么数据、预期看到什么，再展开答案对照。每个答案最后一段是“面试官透露”，就是下一问开头的信息。常见的丢分点：不问就动手；一上来就怪对方或者怪网络；看到原始输出不逐行读；只加重试，把问题盖住；修完不验证。

### 开场：缩小范围

**Q1.** 好的做法是先问能把范围砍掉一半的问题，每个问题背后都对应一个假设：

- 什么时候开始的？那前后改过什么：发版、配置、基础设施，还有流量模式？
- 所有机器都有，还是某几台？只有这个合作方，还是所有外部接口都有？
- 有没有时间规律：高峰、低谷、整点？
- 失败长什么样：请求发出去多久之后报错？立刻报错和等很久才超时，是两类完全不同的问题。
- 失败发生在新建的连接上，还是从连接池里复用的旧连接上？请同事在日志里加上“这个连接上一次使用到现在隔了多久”，按它分组统计失败率。
- 流量怎么出去：直连、经过 NAT、代理还是防火墙？
- 合作方那边能不能查到失败的请求？他们的负载均衡器空闲超时是多少？

可以顺口说明一点：“Connection reset by peer” 只说明本机收到了一个 RST（reset，重置）报文，不说明是对端服务器发的；路径上的设备也可以冒用对端的地址发 RST。

丢分点：直接说“是合作方的问题”；还没缩小范围就要求在所有机器上抓包；先把重试次数调大，把问题藏起来。

面试官透露：Q2 开头的四条情况，以及按空闲时长分组的统计和 dig 的结果。

**Q2.** 先读表：空闲不到 350 秒的 1,265,800 个请求一个都没失败，空闲 350 秒及以上的 3,800 个请求全部失败。一共 1,269,600 个请求、3,800 个失败，失败率 0.30%，正是报上来的那个数。分界线这么干净，说明有一个“空闲 350 秒左右就失效”的东西。

再逐个看假设：

- 合作方服务器的 keep-alive 超时：服务器主动关闭空闲连接时会发 FIN，正常关闭；客户端库在复用之前就能发现连接已经关了，不会把请求发出去。而且常见的默认值也不是 350 秒（例如 nginx 的 `keepalive_timeout` 默认 75 秒），合作方的负载均衡器是 1 小时，合作方也查不到这些请求。不符合。
- 我们这边路径上有设备（NAT、防火墙、负载均衡器）的空闲超时在 350 秒左右：设备删掉了这条连接的状态，之后再有包过来，它就回 RST。符合：分界线干净；请求一发出就失败；合作方什么也没收到；夜里流量低，连接容易一闲就是很久。
- DNS 变了，旧连接还连着已经下线的 IP：dig 显示两个 IP 都在，TTL 60 秒；而且这种问题不会按空闲时长整整齐齐地切开。不符合。
- 对方过载或者限流：应该出现在高峰时段，而且新连接也会失败。不符合。

为什么三周前才开始：以前夜间批处理一直在用这些连接，没有哪个连接能闲到 350 秒；批处理搬走以后，夜里的连接经常一闲就是好几分钟。

下一步：在一台机器上抓包，看 RST 从哪里来、什么时候来、空闲期间有没有任何包。

丢分点：看了这张表还在讨论 DNS；没有把“三周前”和“批处理搬走”联系起来。

面试官透露：一条失败连接的抓包（Q3）。

### 抓包：谁发的 RST

**Q3.** 逐行读：

- 前三行是三次握手。SYN 发出到收到 SYN-ACK 用了 38.156 毫秒，这就是到合作方的往返时间（RTT），大约 38 毫秒。
- 后半段前三行是这条连接上一次正常使用：发请求、收响应、客户端确认。相对序号从 598 开始，是因为省略掉的 TLS 握手已经用掉了前面的字节。
- 02:14:07.482913，客户端在 413.081 秒的空闲之后发出新请求。空闲期间两边没有任何包：没有 FIN，也没有 keepalive 探测。时间戳选项（TS val）也对得上：客户端的 TS val 从 1193046413 涨到 1193459494，差 413081；Linux 的 TCP 时间戳时钟每毫秒加 1，正好是 413.081 秒。
- 最后一行，RST 在请求发出 0.416 毫秒之后就到了。合作方在 38 毫秒之外，回复不可能这么快；这个 RST 只能来自离本机很近的设备，它冒用了合作方的地址。
- RST 的序号显示的是绝对值 3772195816（tcpdump 对不带 ACK 标志的包不换算成相对序号）。服务器的初始序号是 3772190654，加 5162 正好是 3772195816，也就是客户端已经确认到的位置（`ack 5162`）。RFC 9293 规定：如果连接不存在，收到一个带 ACK 的报文，就回复 `<SEQ=SEG.ACK><CTL=RST>`，即拿对方报文里的确认号当序号回一个 RST。这里正是这个形状：发 RST 的一方手里已经没有这条连接了。

结论：空闲 413 秒之后，路径上一个离本机很近的设备已经忘了这条连接，客户端一发数据，它就回 RST。下一步要问：从这台机器出去，经过哪些设备？

丢分点：只看到“有 RST”就说是合作方发的；不看时间差；不看空闲期间有没有包。

面试官透露：子网的默认路由指向 NAT 网关（Q4）。

**Q4.** 三种证据，最好都拿到：

- 查文档：AWS 的 NAT 网关排障文档写着，经过 NAT 网关的连接空闲 350 秒或更久就会超时；超时以后，后面的资源再往这条连接上发包，NAT 网关回 RST（不发 FIN）；解决办法是让连接上有流量，或者在实例上开启间隔小于 350 秒的 TCP keepalive。
- 看指标：NAT 网关的 CloudWatch 指标 `IdleTimeoutCount`，按文档的定义，是没有正常关闭、并且 350 秒没有活动、因而从活跃状态转为空闲的连接数；文档说它上升可能说明 NAT 网关后面的客户端在复用失效的连接。看它和失败数是不是同步，夜里是不是有高峰。
- 做受控实验：在同一台机器上开一个连接，分别空闲 340 秒和 360 秒再发请求，各重复 20 次；再从一台有公网 IP、不经过 NAT 网关的机器做同样的测试，作为对照。请求要选一个无副作用的 GET 接口。

```py
import time
import requests

def idle_then_get(url, idle_s):
    s = requests.Session()
    s.get(url)              # opens one pooled connection
    time.sleep(idle_s)      # leave it idle
    try:
        s.get(url)          # reuses the same connection
        return "ok"
    except requests.ConnectionError as e:
        return f"error: {e}"
```

预测：经过 NAT 网关时，340 秒全部成功，360 秒全部失败；对照机器全部成功。

丢分点：只凭推理就宣布根因；没有对照；实验里每次都新建连接，那就测不到空闲超时了。

面试官透露：实验和指标的结果（Q5 开头），以及同事说“我们开了 keepalive”。

### 解释与修复

**Q5.** 三层原因：

1. 这三个 sysctl 只是默认值，只对设置了 `SO_KEEPALIVE` 选项的套接字生效（见 tcp(7) 手册）。`ss -o` 会给开了 keepalive 的连接显示 `timer:(keepalive,…)`，这里两条连接都没有，说明根本没开。urllib3 默认的套接字选项只有 `TCP_NODELAY`，不包括 `SO_KEEPALIVE`。
2. 就算开了，按这组默认值，连接要空闲 7200 秒才发第一个探测包，远大于 350 秒，根本赶不上。按默认值判定对端已死要 $7200 + 9 \times 75 = 7875$ 秒。
3. urllib3 从连接池取连接时会调用 `is_connection_dropped`，它做的是一次不等待的“套接字现在可读吗”检查：对端发过 FIN 或 RST，套接字就会变成可读，检查就能发现。可 NAT 网关超时的时候什么也不发，套接字看起来完全正常，要等你发出数据才会收到 RST。所以这个检查拦不住。

可以补一句：收到这种 RST，内核计数器 `TcpEstabResets` 会增加（`nstat -az TcpEstabResets` 可以看），以后可以拿来做监控。

丢分点：看到 sysctl 就以为开了 keepalive；不知道默认要 7200 秒；以为连接池的检查能发现中间设备的超时。

面试官透露：客户端是 Python requests，全局 Session，默认 HTTPAdapter，GET 和 POST 都有（Q6 开头）。

**Q6.** 今晚的改动：给这个 Session 挂一个开启 keepalive 的 adapter，让空闲的连接每 120 秒（小于 350 秒，留足余量）发一次探测。探测包经过 NAT 网关，会刷新它的空闲计时。

```py
import socket

import requests
from requests.adapters import HTTPAdapter
from urllib3.connection import HTTPConnection

KEEPALIVE_OPTS = HTTPConnection.default_socket_options + [
    (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1),
    (socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 120),   # first probe after 120 s idle
    (socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 30),   # then every 30 s without a reply
    (socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3),      # give up after 3 unanswered probes
]

class KeepAliveAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        kwargs["socket_options"] = KEEPALIVE_OPTS
        super().init_poolmanager(*args, **kwargs)

session = requests.Session()
session.mount("https://", KeepAliveAdapter())
```

`TCP_KEEPIDLE`、`TCP_KEEPINTVL`、`TCP_KEEPCNT` 是 Linux 上的选项名，macOS 等系统的名字和支持情况不完全一样。

还要处理重试：

- 只自动重试幂等的请求，比如查询订单状态的 GET。urllib3 的 `Retry` 默认允许重试的方法里本来就没有 POST。
- 创建支付的 POST 不能盲目重试，否则可能扣两次钱。要么用合作方支持的幂等键（idempotency key），要么先查状态再决定。这个场景里 RST 发生在请求到达合作方之前，但客户端一般没法确定这一点，不能拿它当前提。

验证：

- 在机器上跑 `ss -tno state established dst 203.0.113.50`，每条连接都应该带 `timer:(keepalive,…)`，剩余时间不超过两分钟；
- 第二天看按空闲时长分组的失败率，350 秒以上的两组应该降到 0；
- `IdleTimeoutCount` 降到接近 0。

另一种办法是让客户端自己丢掉空闲太久的连接（比如空闲超过 300 秒就关掉重连），效果一样，但 requests 和 urllib3 没有现成的参数，要自己包一层。另外，keepalive 会让连接活得更久；NAT 网关的每个 IPv4 地址到同一个目的地（目的 IP、端口和协议都相同）最多支持 55,000 个并发连接，连接池不要开得太大。

丢分点：只加重试，不加 keepalive；对 POST 也自动重试；改完不验证。

面试官透露：上线后的结果（Q7 开头）。

**Q7.** 长期的原则：一条连接经过的每一层都有自己的空闲超时。客户端连接池的空闲上限要小于服务器的 keep-alive 超时，也要小于路径上所有中间设备的空闲超时；后一条做不到，就让 keepalive 的间隔小于路径上最短的那个空闲超时。具体做法：

- 把这组设置放进公司统一的 HTTP 客户端库，不靠每个服务自己记得；
- 把每个环境的路径上有哪些设备、空闲超时是多少写进文档；
- 客户端必须设置读超时（read timeout），不要无限等。

不能照搬 350 秒，这个数随厂商和配置变化。下面是写这道题时查到的官方文档里的默认值：

| 设备 | 空闲超时 | 超时以后 |
|---|---|---|
| AWS NAT 网关 | 350 秒；文档里没有找到可以调整的选项 | 再有包过来时回 RST |
| AWS 网络负载均衡器（NLB），TCP 监听器 | 默认 350 秒，可设 60–6000 秒 | 再有包过来时回 RST |
| Azure 负载均衡器 | 默认 4 分钟；负载均衡规则和入站 NAT 规则可设 4–100 分钟，出站规则可设 4–120 分钟 | 默认悄悄丢弃，可以开启“TCP 重置”选项 |
| Google Cloud NAT | 已建立的 TCP 连接默认 1200 秒，可调 | — |
| Linux conntrack（自建 NAT 或防火墙） | `nf_conntrack_tcp_timeout_established` 默认 432000 秒（5 天） | 看防火墙规则怎么处理 |
| RFC 5382 对 NAT 的要求 | 已建立连接的空闲超时不得短于 2 小时 4 分钟 | — |

不同语言的默认值也不一样：Go 的 `http.DefaultTransport` 默认开启 30 秒的 TCP keepalive，空闲连接 90 秒后关闭，同样的路径不会出这个问题；Python requests 两样都没有。

悄悄丢包的变体（例如 Azure 负载均衡器的默认行为）更难查：客户端不会立刻报错，而是卡住。请求发出去没有回应，内核按指数退避一次次重传，直到 `tcp_retries2`（默认 15）用完才放弃；内核文档说这对应的理论超时是 924.6 秒，而且只是下限。没有设读超时的话，应用里看到的就是请求挂了十几分钟。对策：客户端一定要设读超时；需要时用 `TCP_USER_TIMEOUT` 限制已发出的数据最多等多久没有确认；keepalive 照样要开。

丢分点：把 350 秒当成普遍规律；只修这一个服务；没想到悄悄丢包的情况。

面试官透露：问题已经解决，进入复盘（Q8）。

### 复盘

**Q8.** 从监控、测试和变更流程三方面说：

- 监控：按连接空闲时长分组的失败率，这次的关键证据就是这张表；NAT 网关的 `IdleTimeoutCount` 告警；主机上的 `TcpEstabResets` 增长；按错误类型分开统计，不要把连接被重置和普通的 5xx 混在一起。
- 主动探测：一个定时任务，故意让连接空闲比如 400 秒再发请求，空闲超时一变就能发现。
- 测试：压测或者长时间稳定性测试（soak test）里加一段“流量停几分钟再恢复”；只在持续的高流量下测，永远测不出空闲超时。
- 日志：客户端记录每次请求用的是新连接还是复用的连接，复用前空闲了多久。
- 变更流程：流量模式的变化（比如搬走夜间批处理）也是变更，要检查它会不会让连接空闲得更久；每加一层网络设备，就把它的空闲超时记下来。
- 别让重试掩盖问题：重试一次就能成功的错误最容易被忽略，要单独统计“靠重试才成功”的比例并设阈值。

丢分点：只说“加监控”，不说监控什么；只讲这一个服务的修复，不讲流程。

<details>
<summary>验证代码（可运行）</summary>

```python
import re
import socket

capture = """\
02:07:14.201455 IP 10.0.12.34.48316 > 203.0.113.50.443: Flags [S], seq 1946283124, win 64240, options [mss 1460,sackOK,TS val 1193046212 ecr 0,nop,wscale 7], length 0
02:07:14.239611 IP 203.0.113.50.443 > 10.0.12.34.48316: Flags [S.], seq 3772190654, ack 1946283125, win 65160, options [mss 1460,sackOK,TS val 3320158872 ecr 1193046212,nop,wscale 7], length 0
02:07:14.239652 IP 10.0.12.34.48316 > 203.0.113.50.443: Flags [.], ack 1, win 502, options [nop,nop,TS val 1193046250 ecr 3320158872], length 0
02:07:14.363504 IP 10.0.12.34.48316 > 203.0.113.50.443: Flags [P.], seq 598:1121, ack 4467, win 501, options [nop,nop,TS val 1193046374 ecr 3320158958], length 523
02:07:14.401871 IP 203.0.113.50.443 > 10.0.12.34.48316: Flags [P.], seq 4467:5162, ack 1121, win 501, options [nop,nop,TS val 3320159034 ecr 1193046374], length 695
02:07:14.401899 IP 10.0.12.34.48316 > 203.0.113.50.443: Flags [.], ack 5162, win 497, options [nop,nop,TS val 1193046413 ecr 3320159034], length 0
02:14:07.482913 IP 10.0.12.34.48316 > 203.0.113.50.443: Flags [P.], seq 1121:1533, ack 5162, win 497, options [nop,nop,TS val 1193459494 ecr 3320159034], length 412
02:14:07.483329 IP 203.0.113.50.443 > 10.0.12.34.48316: Flags [R], seq 3772195816, win 0, length 0
"""

def parse(line):
    hms, _, src, _, dst = line.split()[:5]
    h, m, s = hms.split(":")
    p = {"t": int(h) * 3600 + int(m) * 60 + float(s), "src": src, "dst": dst.rstrip(":"),
         "flags": re.search(r"Flags \[([^\]]*)\]", line).group(1),
         "len": int(re.search(r"length (\d+)", line).group(1))}
    if (mo := re.search(r"seq (\d+)(?::(\d+))?", line)):
        p["seq"] = int(mo.group(1)); p["seq_end"] = int(mo.group(2)) if mo.group(2) else None
    if (mo := re.search(r"ack (\d+)", line)):
        p["ack"] = int(mo.group(1))
    if (mo := re.search(r"TS val (\d+)", line)):
        p["tsval"] = int(mo.group(1))
    return p

pk = [parse(l) for l in capture.splitlines()]
client = "10.0.12.34.48316"
syn, synack, ack, req1, resp1, ack1, req2, rst = pk

# handshake RTT and the reset delay
assert round((synack["t"] - syn["t"]) * 1000, 3) == 38.156
assert synack["ack"] == syn["seq"] + 1
assert round((rst["t"] - req2["t"]) * 1000, 3) == 0.416
assert rst["flags"] == "R" and rst["src"] != client

# idle gap: wall clock and TCP timestamps (Linux TS clock = 1 ms per tick)
idle = req2["t"] - ack1["t"]
assert round(idle, 3) == 413.081
assert req2["tsval"] - ack1["tsval"] == 413081
client_pk = [p for p in pk if p["src"] == client]
for p in client_pk:
    dt_ms = (p["t"] - client_pk[0]["t"]) * 1000
    assert abs((p["tsval"] - client_pk[0]["tsval"]) - dt_ms) < 1.0, p

# relative sequence numbers and lengths are consistent
for p in (req1, resp1, req2):
    assert p["seq_end"] - p["seq"] == p["len"]
assert ack1["ack"] == resp1["seq_end"] == 5162
assert req2["seq"] == req1["seq_end"]

# the bare RST carries an absolute seq = server ISN + the client's relative ack
server_isn = synack["seq"]
assert server_isn + req2["ack"] == rst["seq"] == 3772195816

# failure table by idle age
table = [(1_180_000, 0), (76_500, 0), (9_300, 0), (2_940, 2_940), (860, 860)]
total = sum(r for r, _ in table)
fails = sum(f for _, f in table)
assert total == 1_269_600 and fails == 3_800
assert sum(r for r, _ in table[:3]) == 1_265_800
assert round(fails / total * 100, 2) == 0.30

# keepalive arithmetic: Linux defaults vs. the 350 s NAT idle timeout
ka_time, ka_intvl, ka_probes = 7200, 75, 9
assert ka_time > 350
assert ka_time + ka_probes * ka_intvl == 7875
assert 120 < 350 and 120 + 3 * 30 == 210

# tcp_retries2 = 15 model timeout (kernel tcp_model_timeout with TCP_RTO_MIN = 200 ms, TCP_RTO_MAX = 120 s)
def tcp_model_timeout_ms(boundary, rto_base=200, rto_max=120_000):
    thresh = (rto_max // rto_base).bit_length() - 1      # ilog2
    if boundary <= thresh:
        return ((2 << boundary) - 1) * rto_base
    return ((2 << thresh) - 1) * rto_base + (boundary - thresh) * rto_max
assert tcp_model_timeout_ms(15) == 924_600

# the socket options from Q6 really land on a socket (loopback only, Linux names)
opts = [(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1), (socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 120),
        (socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 30), (socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3)]
srv = socket.socket(); srv.bind(("127.0.0.1", 0)); srv.listen(1)
plain = socket.create_connection(srv.getsockname())
assert plain.getsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE) == 0    # off unless the app asks
cli = socket.socket()
for level, name, value in opts:
    cli.setsockopt(level, name, value)
cli.connect(srv.getsockname())
for level, name, value in opts:
    assert cli.getsockopt(level, name) == value
for s in (plain, cli, srv):
    s.close()
print("network-debugging-roleplay-a checks ok")
```

</details>

</details>
