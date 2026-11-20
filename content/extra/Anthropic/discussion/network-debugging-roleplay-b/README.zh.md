# 网络排障角色扮演 B：小文件能下，大文件卡住

## 题目

面试官扮演值班同事，手里有日志、监控和机器权限；你来主导排查：提问、要数据、让他跑命令、给出判断。每一步的题目里写着上一步面试官透露的信息，可以一步一步往下练；场景、数字和命令输出都是我们编的，输出格式按真实工具核对过，数值不是实测。

### 开场：缩小范围

**Q1.** 同事说：“DC1 机房的训练机 trainer-a（10.10.0.21）要从 DC2 机房的制品服务器 `artifacts.dc2.internal`（10.20.0.15，端口 8080）下载模型文件，这几天一直卡住，最后超时。可是服务的健康检查一直是绿的。”他在 trainer-a 上跑了两条命令：

```text
$ curl -sSv -o /dev/null --max-time 20 http://artifacts.dc2.internal:8080/models/ckpt-0042.bin
*   Trying 10.20.0.15:8080...
* Connected to artifacts.dc2.internal (10.20.0.15) port 8080 (#0)
> GET /models/ckpt-0042.bin HTTP/1.1
> Host: artifacts.dc2.internal:8080
> User-Agent: curl/7.81.0
> Accept: */*
> 
* Mark bundle as not supporting multiuse
< HTTP/1.1 200 OK
< Server: depot
< Content-Type: application/octet-stream
< Content-Length: 5242880
< Connection: keep-alive
< 
* Operation timed out after 20001 milliseconds with 0 out of 5242880 bytes received
* Closing connection 0
curl: (28) Operation timed out after 20001 milliseconds with 0 out of 5242880 bytes received
$ curl -sSv -o /dev/null --max-time 20 http://artifacts.dc2.internal:8080/healthz
*   Trying 10.20.0.15:8080...
* Connected to artifacts.dc2.internal (10.20.0.15) port 8080 (#0)
> GET /healthz HTTP/1.1
> Host: artifacts.dc2.internal:8080
> User-Agent: curl/7.81.0
> Accept: */*
> 
* Mark bundle as not supporting multiuse
< HTTP/1.1 200 OK
< Server: depot
< Content-Type: text/plain
< Content-Length: 2
< Connection: keep-alive
< 
{ [2 bytes data]
* Connection #0 to host artifacts.dc2.internal left intact
```

你从这两段输出里读出什么？接下来问什么？

**Q2.** 你问到的情况：(1) 上周四，两个机房之间的专线换成了经过路由器的隧道，两边路由器 rtr-dc1、rtr-dc2 上隧道接口 `tun0` 的 MTU 是 1400，服务器的网卡 MTU 都是 1500，没有做别的调整；(2) 从那以后，跨机房下载大文件都会卡住，DC2 机房内部下载同一个文件没问题；(3) 从 trainer-a 往 DC2 上传大文件也会卡住，跨机房的小请求都正常。同事还在 trainer-a 上跑了 ping 和 mtr：

```text
$ ping -c 3 10.20.0.15
PING 10.20.0.15 (10.20.0.15) 56(84) bytes of data.
64 bytes from 10.20.0.15: icmp_seq=1 ttl=62 time=12.3 ms
64 bytes from 10.20.0.15: icmp_seq=2 ttl=62 time=12.1 ms
64 bytes from 10.20.0.15: icmp_seq=3 ttl=62 time=12.2 ms

--- 10.20.0.15 ping statistics ---
3 packets transmitted, 3 received, 0% packet loss, time 2003ms
rtt min/avg/max/mdev = 12.100/12.200/12.300/0.081 ms
$ mtr -r -n -c 10 10.20.0.15
Start: 2026-11-16T10:35:12+0000
HOST: trainer-a                   Loss%   Snt   Last   Avg  Best  Wrst StDev
  1.|-- 10.10.0.1                  0.0%    10    0.3   0.3   0.2   0.5   0.1
  2.|-- 172.31.255.2               0.0%    10   11.9  12.0  11.8  12.4   0.2
  3.|-- 10.20.0.15                 0.0%    10   12.2  12.2  12.0  12.5   0.1
```

同事说：“网络没丢包，延迟也正常，应该不是网络的问题吧？”你怎么看？有哪些假设，下一步做什么？

### 定位：包卡在哪里

**Q3.** 同事按你说的，在 trainer-a 上跑了两组禁止分片（DF 位）的 ping：

```text
$ ping -M do -s 1472 -c 3 -W 1 10.20.0.15
PING 10.20.0.15 (10.20.0.15) 1472(1500) bytes of data.

--- 10.20.0.15 ping statistics ---
3 packets transmitted, 0 received, 100% packet loss, time 2041ms
$ ping -M do -s 1372 -c 3 -W 1 10.20.0.15
PING 10.20.0.15 (10.20.0.15) 1372(1400) bytes of data.
1380 bytes from 10.20.0.15: icmp_seq=1 ttl=62 time=12.4 ms
1380 bytes from 10.20.0.15: icmp_seq=2 ttl=62 time=12.3 ms
1380 bytes from 10.20.0.15: icmp_seq=3 ttl=62 time=12.4 ms

--- 10.20.0.15 ping statistics ---
3 packets transmitted, 3 received, 0% packet loss, time 2003ms
rtt min/avg/max/mdev = 12.300/12.366/12.400/0.047 ms
```

这两组结果说明什么？第一组的输出里还有什么值得注意的地方？

**Q4.** 同事在服务器 10.20.0.15 上开始抓包：`sudo tcpdump -nn -i eth0 -w dl.pcap '(host 10.10.0.21 and tcp port 8080) or icmp'`，然后在 trainer-a 上再跑一次下载，约 15 秒后停止抓包。下面是抓到的全部内容：

```text
$ TZ=UTC tcpdump -nn -r dl.pcap
10:41:03.120114 IP 10.10.0.21.40112 > 10.20.0.15.8080: Flags [S], seq 3117402113, win 64240, options [mss 1460,sackOK,TS val 811220344 ecr 0,nop,wscale 7], length 0
10:41:03.120152 IP 10.20.0.15.8080 > 10.10.0.21.40112: Flags [S.], seq 2079513358, ack 3117402114, win 65160, options [mss 1460,sackOK,TS val 2946114771 ecr 811220344,nop,wscale 7], length 0
10:41:03.131950 IP 10.10.0.21.40112 > 10.20.0.15.8080: Flags [.], ack 1, win 502, options [nop,nop,TS val 811220356 ecr 2946114771], length 0
10:41:03.132015 IP 10.10.0.21.40112 > 10.20.0.15.8080: Flags [P.], seq 1:112, ack 1, win 502, options [nop,nop,TS val 811220356 ecr 2946114771], length 111: HTTP: GET /models/ckpt-0042.bin HTTP/1.1
10:41:03.132031 IP 10.20.0.15.8080 > 10.10.0.21.40112: Flags [.], ack 112, win 509, options [nop,nop,TS val 2946114782 ecr 811220356], length 0
10:41:03.132572 IP 10.20.0.15.8080 > 10.10.0.21.40112: Flags [P.], seq 1:124, ack 112, win 509, options [nop,nop,TS val 2946114783 ecr 811220356], length 123: HTTP: HTTP/1.1 200 OK
10:41:03.132618 IP 10.20.0.15.8080 > 10.10.0.21.40112: Flags [.], seq 124:1572, ack 112, win 509, options [nop,nop,TS val 2946114783 ecr 811220356], length 1448: HTTP
10:41:03.132631 IP 10.20.0.15.8080 > 10.10.0.21.40112: Flags [.], seq 1572:3020, ack 112, win 509, options [nop,nop,TS val 2946114783 ecr 811220356], length 1448: HTTP
10:41:03.132644 IP 10.20.0.15.8080 > 10.10.0.21.40112: Flags [.], seq 3020:4468, ack 112, win 509, options [nop,nop,TS val 2946114783 ecr 811220356], length 1448: HTTP
10:41:03.132657 IP 10.20.0.15.8080 > 10.10.0.21.40112: Flags [.], seq 4468:5916, ack 112, win 509, options [nop,nop,TS val 2946114783 ecr 811220356], length 1448: HTTP
10:41:03.132670 IP 10.20.0.15.8080 > 10.10.0.21.40112: Flags [.], seq 5916:7364, ack 112, win 509, options [nop,nop,TS val 2946114783 ecr 811220356], length 1448: HTTP
10:41:03.132683 IP 10.20.0.15.8080 > 10.10.0.21.40112: Flags [.], seq 7364:8812, ack 112, win 509, options [nop,nop,TS val 2946114783 ecr 811220356], length 1448: HTTP
10:41:03.132696 IP 10.20.0.15.8080 > 10.10.0.21.40112: Flags [.], seq 8812:10260, ack 112, win 509, options [nop,nop,TS val 2946114783 ecr 811220356], length 1448: HTTP
10:41:03.132709 IP 10.20.0.15.8080 > 10.10.0.21.40112: Flags [.], seq 10260:11708, ack 112, win 509, options [nop,nop,TS val 2946114783 ecr 811220356], length 1448: HTTP
10:41:03.132722 IP 10.20.0.15.8080 > 10.10.0.21.40112: Flags [.], seq 11708:13156, ack 112, win 509, options [nop,nop,TS val 2946114783 ecr 811220356], length 1448: HTTP
10:41:03.144421 IP 10.10.0.21.40112 > 10.20.0.15.8080: Flags [.], ack 124, win 501, options [nop,nop,TS val 811220368 ecr 2946114783], length 0
10:41:03.144446 IP 10.20.0.15.8080 > 10.10.0.21.40112: Flags [.], seq 13156:14604, ack 112, win 509, options [nop,nop,TS val 2946114795 ecr 811220368], length 1448: HTTP
10:41:03.144459 IP 10.20.0.15.8080 > 10.10.0.21.40112: Flags [.], seq 14604:16052, ack 112, win 509, options [nop,nop,TS val 2946114795 ecr 811220368], length 1448: HTTP
10:41:03.177842 IP 10.20.0.15.8080 > 10.10.0.21.40112: Flags [.], seq 16052:17500, ack 112, win 509, options [nop,nop,TS val 2946114828 ecr 811220368], length 1448: HTTP
10:41:03.389842 IP 10.20.0.15.8080 > 10.10.0.21.40112: Flags [.], seq 124:1572, ack 112, win 509, options [nop,nop,TS val 2946115040 ecr 811220368], length 1448: HTTP
10:41:03.813842 IP 10.20.0.15.8080 > 10.10.0.21.40112: Flags [.], seq 124:1572, ack 112, win 509, options [nop,nop,TS val 2946115464 ecr 811220368], length 1448: HTTP
10:41:04.661842 IP 10.20.0.15.8080 > 10.10.0.21.40112: Flags [.], seq 124:1572, ack 112, win 509, options [nop,nop,TS val 2946116312 ecr 811220368], length 1448: HTTP
10:41:06.357842 IP 10.20.0.15.8080 > 10.10.0.21.40112: Flags [.], seq 124:1572, ack 112, win 509, options [nop,nop,TS val 2946118008 ecr 811220368], length 1448: HTTP
10:41:09.749842 IP 10.20.0.15.8080 > 10.10.0.21.40112: Flags [.], seq 124:1572, ack 112, win 509, options [nop,nop,TS val 2946121400 ecr 811220368], length 1448: HTTP
10:41:16.533842 IP 10.20.0.15.8080 > 10.10.0.21.40112: Flags [.], seq 124:1572, ack 112, win 509, options [nop,nop,TS val 2946128184 ecr 811220368], length 1448: HTTP
```

停止抓包前，同事在服务器上看了这条连接的 `ss -tin`（下面只留了相关的字段，其余用 `...` 省略）和路由缓存：

```text
	 cubic wscale:7,7 rto:13568 backoff:6 ... mss:1448 pmtu:1500 ... advmss:1448 cwnd:1 ... unacked:12 retrans:1/6 lost:12 ...
$ ip route get 10.10.0.21
10.10.0.21 via 10.20.0.1 dev eth0 src 10.20.0.15 uid 1000 
    cache 
```

逐行读一下：哪些包到了，哪些没到，服务器在做什么？问题出在哪一类机制上？下一步去哪里看？

### 止血与根治

**Q5.** 同事在 rtr-dc2 朝服务器那一侧的网卡上抓包，trainer-a 又发起了一次下载。下面是截取的两行：

```text
$ TZ=UTC tcpdump -nn -S -r rtr-dc2-lan.pcap
11:26:40.508113 IP 10.20.0.15.8080 > 10.10.0.21.40388: Flags [.], seq 3851142341:3851143789, ack 1405993833, win 509, options [nop,nop,TS val 2948891572 ecr 813997132], length 1448: HTTP
11:26:40.508390 IP 10.20.0.1 > 10.20.0.15: ICMP 10.10.0.21 unreachable - need to frag (mtu 1400), length 556
```

他还查到：rtr-dc2 和服务器网段之间有一台透明模式（bridge）的防火墙，ICMP 只放行回显请求和回显应答（ping 用的）以及超时报文（time exceeded，traceroute 用的），其余 ICMP 一律丢弃；DC1 那边用的是同一套规则模板。改防火墙规则要走评审，最早明天；路由器的配置值班可以直接改，十分钟能上线。根因是什么？现在怎么止血？有几种做法，各有什么代价，你选哪个？

**Q6.** 一个小时后，两边路由器都加了 MSS 钳制（MSS clamping），跨机房下载恢复了。同事在服务器上又抓了一次新连接，下面是开头部分（省略了其余的数据段）：

```text
$ TZ=UTC tcpdump -nn -r after.pcap
12:03:17.402211 IP 10.10.0.21.40566 > 10.20.0.15.8080: Flags [S], seq 2210384467, win 64240, options [mss 1360,sackOK,TS val 815120987 ecr 0,nop,wscale 7], length 0
12:03:17.402248 IP 10.20.0.15.8080 > 10.10.0.21.40566: Flags [S.], seq 1186735390, ack 2210384468, win 65160, options [mss 1460,sackOK,TS val 2950015432 ecr 815120987,nop,wscale 7], length 0
12:03:17.414101 IP 10.10.0.21.40566 > 10.20.0.15.8080: Flags [.], ack 1, win 502, options [nop,nop,TS val 815120999 ecr 2950015432], length 0
12:03:17.414169 IP 10.10.0.21.40566 > 10.20.0.15.8080: Flags [P.], seq 1:112, ack 1, win 502, options [nop,nop,TS val 815120999 ecr 2950015432], length 111: HTTP: GET /models/ckpt-0042.bin HTTP/1.1
12:03:17.414185 IP 10.20.0.15.8080 > 10.10.0.21.40566: Flags [.], ack 112, win 509, options [nop,nop,TS val 2950015443 ecr 815120999], length 0
12:03:17.414702 IP 10.20.0.15.8080 > 10.10.0.21.40566: Flags [P.], seq 1:124, ack 112, win 509, options [nop,nop,TS val 2950015444 ecr 815120999], length 123: HTTP: HTTP/1.1 200 OK
12:03:17.414744 IP 10.20.0.15.8080 > 10.10.0.21.40566: Flags [.], seq 124:1472, ack 112, win 509, options [nop,nop,TS val 2950015444 ecr 815120999], length 1348: HTTP
12:03:17.414756 IP 10.20.0.15.8080 > 10.10.0.21.40566: Flags [.], seq 1472:2820, ack 112, win 509, options [nop,nop,TS val 2950015444 ecr 815120999], length 1348: HTTP
12:03:17.426533 IP 10.10.0.21.40566 > 10.20.0.15.8080: Flags [.], ack 2820, win 501, options [nop,nop,TS val 815121011 ecr 2950015444], length 0
```

读一下这几行，为什么两个方向的 mss 不一样？长期怎么根治？MSS 钳制管不到哪些情况？防火墙改完以后怎么验证？

### 复盘

**Q7.** 面试官最后问：“隧道上线以后，问题拖了好几天才有人报上来，健康检查一直是绿的。怎样才能更早发现，或者在换隧道的时候就避免？”

## 参考解答

<details>
<summary>展开参考解答</summary>

这种轮次的证据很少：流程页只说性能和基础设施岗位的专项轮里有“网络排障角色扮演”，一位性能工程候选人的候选人门户里也列了它；没有人记录过场景、时长和评分方式。下面按排障角色扮演的一般做法来写。面试官大概率看这几件事：会不会主动问对问题；每一步要的数据能不能区分假设；读不读得懂原始输出（curl、ping、tcpdump、ss）；先止血再根治，并说清楚每种做法的代价；最后能不能复盘。

练习方法：只看题目，先自己说出这一步要问什么、要什么数据、预期看到什么，再展开答案对照。每个答案最后一段是“面试官透露”，就是下一问开头的信息。常见的丢分点：看到 ping 正常就排除网络；看到原始输出不逐行读；不问最近改了什么；止血的办法说不出代价；修完不验证。

### 开场：缩小范围

**Q1.** 先把 curl 的输出一行行读完，它已经排除了很多东西：

- DNS 解析正常，TCP 连接建立了（`Connected to`），请求发出去了，服务器的响应头也收到了：`200 OK`，`Content-Length: 5242880`（5 MiB）。
- 但响应体一个字节都没收到（`0 out of 5242880 bytes received`），20 秒后超时。不是“慢”，是“完全不动”。
- `/healthz` 的响应只有两个字节，整个请求很快就完成了。

所以服务本身是活的，连接也通。小的数据能过，一开始传大块数据就卡死，这很像“大的包过不去”。服务器慢或者磁盘慢一般是慢慢传，不会正好卡在零字节。

接下来问：

- 从什么时候开始的？那前后网络有没有变化：换线路、加隧道、加防火墙、改 MTU？
- 只有这台机器，还是跨机房的所有机器？机房内部下载同一个文件行不行？
- 反方向（从 trainer-a 往 DC2 传大文件）行不行？
- 小文件行不行，大概多大开始不行？
- 健康检查是从哪里发起的，它传的数据有多大？

丢分点：看到健康检查是绿的就认定服务器没问题、去查客户端；不读 curl 输出里的字节数；一上来就重启服务。

面试官透露：Q2 开头的情况，以及 ping 和 mtr 的结果。

**Q2.** 不同意“不是网络的问题”。ping 和 mtr 用的都是小包：ping 默认 56 字节数据，加上 ICMP 头和 IP 头一共 84 字节；mtr 默认的探测包是 64 字节。它们只能说明小包能走通，说明不了 1500 字节的包能不能过。

从输出里能读出的东西：ttl=62，说明回包经过了两个路由器（Linux 默认的初始 TTL 是 64，减了 2）；mtr 的三跳是 rtr-dc1 的内网地址、rtr-dc2 的隧道地址、服务器本身，和隧道的拓扑一致；往返时间约 12 毫秒，没有丢包。

假设：

1. 路径 MTU（path MTU）问题：隧道接口的 MTU 只有 1400，服务器发的是 1500 字节、带 DF 位（不许分片）的包，进不了隧道。正常情况下，路由器会回一个 ICMP“需要分片”报文，告诉发送方把包改小；如果这个报文被谁丢了，发送方就永远不知道，一直重发同样大小的包。这叫 PMTUD 黑洞（black hole）。它符合所有现象：上周四换隧道以后开始，机房内部正常，两个方向都卡，小请求正常，ping 正常。
2. 路径上有设备（防火墙、入侵防御系统）按内容或者大小丢包。有可能，但时间点和隧道上线对得上，先查第一个。
3. 窗口或缓冲区配置问题：那样会很慢，不会一个字节都收不到。不符合。
4. 服务器或磁盘问题：机房内部下载正常。不符合。
5. 隧道丢包：ping 和 mtr 没有丢包，而且丢包会让传输变慢，不会每次都卡在同一个地方。不符合。

下一步：用带 DF 位的 ping 测大包，二分出路径 MTU。`-s` 是 ICMP 数据的字节数，加上 8 字节 ICMP 头和 20 字节 IP 头才是 IP 包的大小：`-s 1472` 对应 1500 字节，`-s 1372` 对应 1400 字节。`-M do` 表示设置 DF 位，并且本机也不分片。

丢分点：同意“不是网络问题”；不知道 ping 和 mtr 默认用的是小包；没有把“上周四换隧道”和现象联系起来。

面试官透露：两组 DF ping 的结果（Q3）。

### 定位：包卡在哪里

**Q3.** 1500 字节的包全丢，1400 字节的包全通，说明路径 MTU 在 1400 到 1499 之间，和隧道接口的 MTU 1400 吻合。要精确的值，可以继续二分，比如试 `-s 1373`（1401 字节）。成功的那组回包显示 `1380 bytes`，是 ICMP 回显应答的大小（1372 字节数据加 8 字节 ICMP 头），不含 IP 头。

第一组输出里更值得注意的是“什么都没打印”。如果 rtr-dc1 回的 ICMP“需要分片”报文到了 trainer-a，ping 会打印一行类似 `From 10.10.0.1 icmp_seq=1 Frag needed and DF set (mtu = 1400)` 的错误，内核也会记住这条路径的 MTU，之后的包就会在本机报错。现在什么都没有，说明这个 ICMP 报文要么没发出来，要么在半路被丢了。这正是黑洞的特征，也解释了上传方向为什么卡住。

下一步回到报告的问题，也就是下载方向。在服务器 10.20.0.15 上抓包，过滤条件一定要包括 `icmp`，否则“没看到 ICMP”就没有意义；同时看这条连接的 `ss -tin`（内核眼里的 MSS、路径 MTU、重传状态）和 `ip route get 10.10.0.21`（内核有没有为这个目的地缓存一个更小的 MTU）。

丢分点：只说“MTU 有问题”，不注意到没有 ICMP 报错这个关键信息；抓包时只抓 TCP 端口，把 ICMP 过滤掉了。

面试官透露：服务器上的抓包、`ss` 和路由缓存（Q4）。

**Q4.** 逐段读：

- 前三行是三次握手。两边的 SYN 里都写着 `mss 1460`，也就是双方都以为路径 MTU 是 1500。客户端回 ACK 用了 11.798 毫秒，这是往返时间。
- 第 4、5 行：请求 111 字节，加上 20 字节 TCP 头、12 字节时间戳选项和 20 字节 IP 头，IP 包 163 字节，到了，服务器确认。
- 第 6 行：响应头 123 字节，IP 包 175 字节。客户端在 10:41:03.144421 回了 `ack 124`，说明它也到了。这就是 curl 能打印出响应头的原因。
- 第 7–15 行：9 个满载的数据段，每段 1448 字节，加上 52 字节头部正好 1500 字节。加上响应头那个包，一共 10 个包，就是初始拥塞窗口。
- 收到 `ack 124` 以后，服务器又发了 2 个新段（慢启动里每确认一个包，窗口加一）。
- 10:41:03.177842 又发了一个新段：这是尾部丢失探测（tail loss probe，TLP），大约两个往返时间收不到确认，就再发一个段试探。确切的等待时间和内核版本、定时器精度有关。
- 从 10:41:03.389842 开始，一直重传第一个数据段 `124:1572`：第一次在 TLP 之后 212 毫秒，这是重传超时（RTO）。Linux 的 RTO 等于平滑往返时间加一个偏差项，偏差项最小 200 毫秒，所以这里约 212 毫秒；之后间隔是 0.424、0.848、1.696、3.392、6.784 秒，每次翻倍，这是指数退避。每次重传的还是 1448 字节，服务器从没把包改小。
- 客户端的确认号一直停在 124：1500 字节的包一个都没到。抓包过滤条件里有 `icmp`，但一个 ICMP 报文都没有。

`ss` 里的字段对得上：`backoff:6` 是已经退避了 6 次，`rto:13568` 正好是 212 × 64 毫秒；`unacked:12` 是 12 个没确认的段（9 + 2 + 1），`lost:12` 是超时以后全部标成丢失，`retrans:1/6` 是当前有 1 个重传在途、累计重传 6 次，`cwnd:1` 是超时后拥塞窗口降到 1。最关键的是 `pmtu:1500` 和路由缓存里没有 `mtu`：内核从来没收到过“需要分片”的消息。

结论：大于 1400 字节的包在路上被丢了，路由器本该回给服务器的 ICMP“需要分片”报文（type 3，code 4）没有到达服务器。这是 PMTUD 黑洞：

- 路径 MTU 发现（PMTUD，RFC 1191）的做法是，发送方给包设置 DF 位；某个路由器发现包太大、又不许分片，就丢掉这个包，并回一个 ICMP 报文，里面带着下一跳的 MTU；发送方收到以后把包改小。Linux 的 TCP 默认就这样做（`net.ipv4.ip_no_pmtu_disc = 0`）。
- 只要这个 ICMP 报文被过滤掉，发送方就会一直按原来的大小重发，连接卡死，直到重传次数用完。RFC 2923 专门描述了这个问题。

还有一个细节：服务器上的 tcpdump 看到的是进入协议栈之前的包，就算本机的 iptables 丢掉 ICMP，tcpdump 也能看到。这里连 tcpdump 都没看到，说明 ICMP 在到达服务器网卡之前就没了，或者根本没发出来。下一步去 rtr-dc2 上抓包，看它有没有发出 ICMP；同时问 rtr-dc2 和服务器之间还有什么设备。

丢分点：看到重传就说“丢包严重”；不注意小包都到了、只有满载的包没到；没发现抓包里没有 ICMP；以为服务器本机的防火墙丢了 ICMP（tcpdump 能看到那种情况）。

面试官透露：rtr-dc2 上的抓包，以及防火墙的规则和改动限制（Q5）。

### 止血与根治

**Q5.** 根因：rtr-dc2 尽职了，0.277 毫秒后就回了“需要分片（mtu 1400）”的 ICMP 报文（`length 556` 是 ICMP 部分的长度，整个 IP 包 576 字节，里面引用了原始包的开头部分），但中间的透明防火墙只放行 ping 和 traceroute 用的 ICMP，把它丢了。DC1 用同一套规则，所以上传方向也一样。ping 和 mtr 正常，是因为它们用的 ICMP 正好在放行名单里，而且包都很小。

马上能做的选项：

1. 在两边路由器上做 MSS 钳制：转发 TCP 的 SYN 和 SYN-ACK 时，把里面的 MSS 选项改小，这样两端一开始就只发放得进隧道的包。

   ```text
   iptables -t mangle -A FORWARD -o tun0 -p tcp --tcp-flags SYN,RST SYN -j TCPMSS --clamp-mss-to-pmtu
   ```

   `--tcp-flags SYN,RST SYN` 匹配 SYN 位为 1、RST 位为 0 的包，SYN 和 SYN-ACK 都包括在内；`--clamp-mss-to-pmtu` 按出接口的路径 MTU 减 40 计算（IPv4），这里是 1360。也可以写死 `--set-mss 1360`。两边都要加：A 发出的 SYN 决定 B 发给 A 的包有多大，B 发出的 SYN-ACK 决定 A 发给 B 的包有多大。iptables-extensions(8) 的 TCPMSS 一节描述的症状正是这个：连接建立以后卡住，一点数据都收不到。代价：只管 TCP；只对新连接生效；把问题盖住了，防火墙的错还在。
2. 在服务器上打开 `net.ipv4.tcp_mtu_probing = 1`（内核默认是 0；有的发行版或云镜像会改，以实际值为准）。设为 1 时，内核怀疑遇到黑洞才启用分组层的 MTU 探测（RFC 4821）。按 Linux 源码，判断条件是从第一次重传开始，经过的时间超过 `tcp_retries1`（默认 3）对应的模型时间，也就是 3 秒。这条连接的 RTO 序列里，要到第 5 次重传那一刻才满足（距第一个数据段约 6.6 秒）。之后 MSS 降到 `tcp_base_mss` 左右（新内核源码里的默认值是 1024，一台 5.19 内核的机器上读出来也是 1024；tcp(7) 手册里写的是 512，以实际读出来的值为准），再慢慢往上探测。代价：每条新连接开头都要卡几秒；只在打开了的机器上生效；只管 TCP。
3. 在服务器上按路由设 MTU，比如 DC2 这边 `ip route add 10.10.0.0/16 via 10.20.0.1 dev eth0 mtu 1400`，DC1 那边反过来再加一条。跨机房的 TCP 和 UDP 都不会超过 1400。不要加 `lock`：按 ip-route(8) 的说明，加了以后内核不再做 PMTUD，IPv4 包不再设置 DF 位。代价：要改每一台机器，新机器容易漏。
4. 把所有服务器网卡的 MTU 改成 1400：影响机房内部的所有流量，改动最大，不推荐。

推荐：现在就在两边路由器上做 MSS 钳制，十分钟能上线，覆盖面最大；明天走评审改防火墙规则，这才是根治。可以顺带在服务器上打开 `tcp_mtu_probing = 1` 作为兜底。

丢分点：只说“把 MTU 调小”，不说调哪里、影响什么；不知道 MSS 钳制只对 TCP 新连接有效；只在一边路由器上做钳制；把止血当成根治，不再改防火墙。

面试官透露：钳制上线后的抓包（Q6）。

**Q6.** 读抓包：

- trainer-a 发出的 SYN 本来写的是 `mss 1460`（它的网卡 MTU 是 1500），经过 rtr-dc1 时被改成了 1400 − 40 = 1360，服务器收到的就是 1360。
- 服务器的 SYN-ACK 写的是 `mss 1460`：抓包点在服务器上，包还没到 rtr-dc2；rtr-dc2 会把它改成 1360，trainer-a 那边看到的是 1360。
- 服务器之后的数据段每段 1348 字节，等于 1360 减 12 字节时间戳选项；加上 52 字节头部，IP 包正好 1400 字节，进得了隧道。11.8 毫秒后客户端确认到 2820，也就是两个段都到了。

根治：

- 改防火墙：放行 ICMP type 3（目的不可达），至少放行 code 4（需要分片）。IPv6 更要放行 ICMPv6 type 2（Packet Too Big）：IPv6 的路由器不做分片，PMTUD 完全靠这个报文，链路 MTU 最小是 1280；哪些 ICMPv6 不该过滤，可以参考 RFC 4890。两个机房的规则模板一起改。
- MSS 钳制留着，作为第二道防线。
- 把隧道的 MTU 和封装开销写进文档。常见封装在外层是 IPv4、底层 MTU 为 1500 时：

| 封装 | 开销（字节） | 隧道 MTU |
|---|---:|---:|
| IPIP | 20 | 1480 |
| GRE（基本头部） | 24 | 1476 |
| VXLAN | 50 | 1450 |
| WireGuard（外层 IPv4） | 60 | 1440 |
| WireGuard（外层 IPv6） | 80 | 1420（WireGuard 接口的默认 MTU） |
| IPsec ESP | 随加密算法、模式和是否 NAT 穿越而变 | 要按实际配置算 |

MSS 钳制管不到的情况：UDP 流量（例如 QUIC、大的 DNS 响应、自己写的 UDP 协议）；钳制上线之前就建立的连接，它们的 MSS 还是 1460，要重连才会好；路径后面如果还有更小的 MTU，路由器按自己出接口算出来的值也不对。

防火墙改完以后的验证：

```text
$ ping -M do -s 1472 -c 3 10.20.0.15
PING 10.20.0.15 (10.20.0.15) 1472(1500) bytes of data.
From 10.10.0.1 icmp_seq=1 Frag needed and DF set (mtu = 1400)
ping: local error: message too long, mtu=1400
ping: local error: message too long, mtu=1400

--- 10.20.0.15 ping statistics ---
3 packets transmitted, 0 received, +3 errors, 100% packet loss, time 2003ms
$ ip route get 10.20.0.15
10.20.0.15 via 10.10.0.1 dev eth0 src 10.10.0.21 uid 1000 
    cache expires 597sec mtu 1400 
```

第一个包收到了 rtr-dc1 的“需要分片”报文，内核把这条路径的 MTU 记成 1400，后面两个包在本机就报错了；路由缓存里有了 `mtu 1400`，600 秒后过期（`net.ipv4.route.mtu_expires` 默认 600），过期后重新发现。在服务器上做同样的检查。也可以用 `tracepath` 看每一跳的路径 MTU。ping 本身不受 MSS 钳制影响，所以这一步证明的是 PMTUD 自己能工作了。ICMP 能到达以后，TCP 收到报文会立刻把 MSS 改小并重传，不用等好几秒的超时。

丢分点：只放行 IPv4 的 ICMP，忘了 ICMPv6；以为钳制以后就不用改防火墙；不知道哪些流量钳制管不到；改完不验证 PMTUD 本身。

面试官透露：问题已经解决，进入复盘（Q7）。

### 复盘

**Q7.** 从变更流程、监控和规则三方面说：

- 变更流程：任何改变 MTU 的变更（加隧道、换封装、加 VPN）上线前后，两个方向都跑一遍带 DF 位的大包 ping 和 `tracepath`，并且实际传一个几 MB 的文件。把“路径 MTU 和 MSS 钳制”放进网络变更的检查清单。
- 健康检查要像真实流量：只返回两个字节的 `/healthz` 永远发现不了大包问题。加一个跨机房的合成探测，定期下载一个几 MB 的文件，检查速度和完整性。
- 监控内核计数器（用 `nstat` 采集）：服务器上的 `TcpExtTCPTimeouts`（重传超时次数）和 `TcpRetransSegs`（重传段数）突然上升；路由器上的 `IcmpOutDestUnreachs` 和 `IcmpMsgOutType3`（发出的目的不可达报文）上升而对端没有反应，就是黑洞的信号；打开 MTU 探测的机器上还可以看 `TcpExtTCPMTUPFail` 和 `TcpExtTCPMTUPSuccess`。
- 防火墙规则检查：规则评审或者自动检查里，禁止“除了 ping 以外的 ICMP 全部丢弃”这类规则，type 3 和 ICMPv6 type 2 必须放行。
- 报告渠道：下载卡住的问题拖了几天才有人报，说明这类故障没有告警；按“请求超时且收到 0 字节响应体”统计，是个便宜又准确的指标。

丢分点：只说“加监控”，不说监控什么；只修这一次，不改变更流程和规则模板。

<details>
<summary>验证代码（可运行）</summary>

```python
import math
import re

IP, TCP, TS_OPT, ICMP = 20, 20, 12, 8

# packet sizes used in the text
assert 1448 + TS_OPT + TCP + IP == 1500
assert 1472 + ICMP + IP == 1500 and 1372 + ICMP + IP == 1400
assert 1372 + ICMP == 1380                      # ping prints the ICMP reply size
assert 56 + ICMP + IP == 84
assert 1373 + ICMP + IP == 1401
assert 1400 - IP - TCP == 1360                  # --clamp-mss-to-pmtu for IPv4
assert 1360 - TS_OPT == 1348 and 1348 + TS_OPT + TCP + IP == 1400
assert 1460 + IP + TCP == 1500 and 1500 - IP - TCP - TS_OPT == 1448

req = b"GET /models/ckpt-0042.bin HTTP/1.1\r\nHost: artifacts.dc2.internal:8080\r\nUser-Agent: curl/7.81.0\r\nAccept: */*\r\n\r\n"
hdr = b"HTTP/1.1 200 OK\r\nServer: depot\r\nContent-Type: application/octet-stream\r\nContent-Length: 5242880\r\nConnection: keep-alive\r\n\r\n"
assert len(req) == 111 and len(hdr) == 123
assert len(req) + TS_OPT + TCP + IP == 163 and len(hdr) + TS_OPT + TCP + IP == 175
assert 5242880 == 5 * 2**20

capture = """\
10:41:03.120114 IP 10.10.0.21.40112 > 10.20.0.15.8080: Flags [S], seq 3117402113, win 64240, options [mss 1460,sackOK,TS val 811220344 ecr 0,nop,wscale 7], length 0
10:41:03.120152 IP 10.20.0.15.8080 > 10.10.0.21.40112: Flags [S.], seq 2079513358, ack 3117402114, win 65160, options [mss 1460,sackOK,TS val 2946114771 ecr 811220344,nop,wscale 7], length 0
10:41:03.131950 IP 10.10.0.21.40112 > 10.20.0.15.8080: Flags [.], ack 1, win 502, options [nop,nop,TS val 811220356 ecr 2946114771], length 0
10:41:03.132015 IP 10.10.0.21.40112 > 10.20.0.15.8080: Flags [P.], seq 1:112, ack 1, win 502, options [nop,nop,TS val 811220356 ecr 2946114771], length 111: HTTP: GET /models/ckpt-0042.bin HTTP/1.1
10:41:03.132031 IP 10.20.0.15.8080 > 10.10.0.21.40112: Flags [.], ack 112, win 509, options [nop,nop,TS val 2946114782 ecr 811220356], length 0
10:41:03.132572 IP 10.20.0.15.8080 > 10.10.0.21.40112: Flags [P.], seq 1:124, ack 112, win 509, options [nop,nop,TS val 2946114783 ecr 811220356], length 123: HTTP: HTTP/1.1 200 OK
10:41:03.132618 IP 10.20.0.15.8080 > 10.10.0.21.40112: Flags [.], seq 124:1572, ack 112, win 509, options [nop,nop,TS val 2946114783 ecr 811220356], length 1448: HTTP
10:41:03.132631 IP 10.20.0.15.8080 > 10.10.0.21.40112: Flags [.], seq 1572:3020, ack 112, win 509, options [nop,nop,TS val 2946114783 ecr 811220356], length 1448: HTTP
10:41:03.132644 IP 10.20.0.15.8080 > 10.10.0.21.40112: Flags [.], seq 3020:4468, ack 112, win 509, options [nop,nop,TS val 2946114783 ecr 811220356], length 1448: HTTP
10:41:03.132657 IP 10.20.0.15.8080 > 10.10.0.21.40112: Flags [.], seq 4468:5916, ack 112, win 509, options [nop,nop,TS val 2946114783 ecr 811220356], length 1448: HTTP
10:41:03.132670 IP 10.20.0.15.8080 > 10.10.0.21.40112: Flags [.], seq 5916:7364, ack 112, win 509, options [nop,nop,TS val 2946114783 ecr 811220356], length 1448: HTTP
10:41:03.132683 IP 10.20.0.15.8080 > 10.10.0.21.40112: Flags [.], seq 7364:8812, ack 112, win 509, options [nop,nop,TS val 2946114783 ecr 811220356], length 1448: HTTP
10:41:03.132696 IP 10.20.0.15.8080 > 10.10.0.21.40112: Flags [.], seq 8812:10260, ack 112, win 509, options [nop,nop,TS val 2946114783 ecr 811220356], length 1448: HTTP
10:41:03.132709 IP 10.20.0.15.8080 > 10.10.0.21.40112: Flags [.], seq 10260:11708, ack 112, win 509, options [nop,nop,TS val 2946114783 ecr 811220356], length 1448: HTTP
10:41:03.132722 IP 10.20.0.15.8080 > 10.10.0.21.40112: Flags [.], seq 11708:13156, ack 112, win 509, options [nop,nop,TS val 2946114783 ecr 811220356], length 1448: HTTP
10:41:03.144421 IP 10.10.0.21.40112 > 10.20.0.15.8080: Flags [.], ack 124, win 501, options [nop,nop,TS val 811220368 ecr 2946114783], length 0
10:41:03.144446 IP 10.20.0.15.8080 > 10.10.0.21.40112: Flags [.], seq 13156:14604, ack 112, win 509, options [nop,nop,TS val 2946114795 ecr 811220368], length 1448: HTTP
10:41:03.144459 IP 10.20.0.15.8080 > 10.10.0.21.40112: Flags [.], seq 14604:16052, ack 112, win 509, options [nop,nop,TS val 2946114795 ecr 811220368], length 1448: HTTP
10:41:03.177842 IP 10.20.0.15.8080 > 10.10.0.21.40112: Flags [.], seq 16052:17500, ack 112, win 509, options [nop,nop,TS val 2946114828 ecr 811220368], length 1448: HTTP
10:41:03.389842 IP 10.20.0.15.8080 > 10.10.0.21.40112: Flags [.], seq 124:1572, ack 112, win 509, options [nop,nop,TS val 2946115040 ecr 811220368], length 1448: HTTP
10:41:03.813842 IP 10.20.0.15.8080 > 10.10.0.21.40112: Flags [.], seq 124:1572, ack 112, win 509, options [nop,nop,TS val 2946115464 ecr 811220368], length 1448: HTTP
10:41:04.661842 IP 10.20.0.15.8080 > 10.10.0.21.40112: Flags [.], seq 124:1572, ack 112, win 509, options [nop,nop,TS val 2946116312 ecr 811220368], length 1448: HTTP
10:41:06.357842 IP 10.20.0.15.8080 > 10.10.0.21.40112: Flags [.], seq 124:1572, ack 112, win 509, options [nop,nop,TS val 2946118008 ecr 811220368], length 1448: HTTP
10:41:09.749842 IP 10.20.0.15.8080 > 10.10.0.21.40112: Flags [.], seq 124:1572, ack 112, win 509, options [nop,nop,TS val 2946121400 ecr 811220368], length 1448: HTTP
10:41:16.533842 IP 10.20.0.15.8080 > 10.10.0.21.40112: Flags [.], seq 124:1572, ack 112, win 509, options [nop,nop,TS val 2946128184 ecr 811220368], length 1448: HTTP
"""

def parse(line):
    hms, _, src, _, dst = line.split()[:5]
    h, m, s = hms.split(":")
    p = {"t": int(h) * 3600 + int(m) * 60 + float(s), "src": src, "dst": dst.rstrip(":"),
         "flags": re.search(r"Flags \[([^\]]*)\]", line).group(1),
         "len": int(re.search(r"length (\d+)", line).group(1)),
         "tsval": int(re.search(r"TS val (\d+)", line).group(1))}
    if (mo := re.search(r", seq (\d+)(?::(\d+))?", line)):
        p["seq"] = int(mo.group(1)); p["seq_end"] = int(mo.group(2)) if mo.group(2) else None
    if (mo := re.search(r", ack (\d+)", line)):
        p["ack"] = int(mo.group(1))
    return p

pk = [parse(l) for l in capture.splitlines()]
assert len(pk) == 25
server, client = "10.20.0.15.8080", "10.10.0.21.40112"
srv = [p for p in pk if p["src"] == server]
cli = [p for p in pk if p["src"] == client]

# RTTs seen from the server
assert round((pk[2]["t"] - pk[1]["t"]) * 1000, 3) == 11.798
hdr_pkt = next(p for p in srv if p["len"] == 123)
hdr_ack = next(p for p in cli if p.get("ack") == 124)
assert round((hdr_ack["t"] - hdr_pkt["t"]) * 1000, 3) == 11.849

# data segments: contiguous 1448-byte segments from 124 to 17500, 12 distinct, never acked
data = [p for p in srv if p["len"] == 1448]
first_tx = {}
for p in data:
    assert p["seq_end"] - p["seq"] == 1448
    first_tx.setdefault(p["seq"], p)
starts = sorted(first_tx)
assert starts[0] == 124 and starts[-1] + 1448 == 17500 and len(starts) == 12
assert all(b - a == 1448 for a, b in zip(starts, starts[1:]))
assert max(p.get("ack", 0) for p in cli) == 124
assert sum(1 for p in srv if p["len"] > 0 and p["t"] < hdr_ack["t"]) == 10   # header + 9 segments = initial window
after_ack = [p for p in data if hdr_ack["t"] < p["t"] < hdr_ack["t"] + 0.001]
assert len(after_ack) == 2

# TLP, RTO and exponential backoff
tlp = first_tx[16052]
last_new_before_tlp = first_tx[14604]
assert round((tlp["t"] - last_new_before_tlp["t"]) * 1000, 3) == 33.383
retx = [p for p in data if p["seq"] == 124][1:]
assert len(retx) == 6
assert round((retx[0]["t"] - tlp["t"]) * 1000, 3) == 212.0
gaps = [round((b["t"] - a["t"]) * 1000, 3) for a, b in zip(retx, retx[1:])]
assert gaps == [212 * 2**k for k in range(1, 6)] == [424, 848, 1696, 3392, 6784]
assert math.ceil(200 + 11.8) == 212 and 212 * 2**6 == 13568      # rto after 6 backoffs
assert len(starts) == 9 + 2 + 1

# timestamps advance with the wall clock (Linux TS clock = 1 ms per tick)
for group in (srv, cli):
    for p in group:
        dt_ms = (p["t"] - group[0]["t"]) * 1000
        assert -1 < dt_ms - (p["tsval"] - group[0]["tsval"]) < 1, p

# black-hole detection with tcp_mtu_probing = 1 (kernel tcp_model_timeout, TCP_RTO_MIN = 200 ms)
def tcp_model_timeout_ms(boundary, rto_base=200, rto_max=120_000):
    thresh = (rto_max // rto_base).bit_length() - 1      # ilog2
    if boundary <= thresh:
        return ((2 << boundary) - 1) * rto_base
    return ((2 << thresh) - 1) * rto_base + (boundary - thresh) * rto_max
assert tcp_model_timeout_ms(3) == 3000 and tcp_model_timeout_ms(15) == 924_600
retrans_stamp = retx[0]["t"]
# tcp_write_timeout runs on each RTO expiry that comes after the first retransmission
fires = [p["t"] for p in retx[1:]]
hit = next(i for i, t in enumerate(fires) if (t - retrans_stamp) * 1000 >= 3000)
assert [round(t - retrans_stamp, 3) for t in fires[:hit + 1]] == [0.424, 1.272, 2.968, 6.36]
assert hit + 2 == 5                                     # the 5th retransmission
assert round(fires[hit] - data[0]["t"], 3) == 6.617

# iputils ping statistics (integer microseconds, same order of operations as ping_common.c)
def ping_stats(times_us):
    n = len(times_us); tsum = sum(times_us); tsum2 = sum(t * t for t in times_us)
    avg = tsum // n
    var = (tsum2 - (tsum * tsum) // n) // n
    mdev = math.isqrt(var)
    f = lambda us: f"{us // 1000}.{us % 1000:03d}"
    return f"rtt min/avg/max/mdev = {f(min(times_us))}/{f(avg)}/{f(max(times_us))}/{f(mdev)} ms"
assert ping_stats([12300, 12100, 12200]) == "rtt min/avg/max/mdev = 12.100/12.200/12.300/0.081 ms"
assert ping_stats([12400, 12300, 12400]) == "rtt min/avg/max/mdev = 12.300/12.366/12.400/0.047 ms"
def ping_time(us):                                           # iputils 20211215, 10 ms <= t < 100 ms
    return f"{(us + 50) // 1000}.{((us + 50) % 1000) // 100}"
assert [ping_time(t) for t in (12300, 12100, 12200, 12400)] == ["12.3", "12.1", "12.2", "12.4"]
assert 64 - 2 == 62                                          # two routers on the way back

# mtr 0.95 report layout (non-wide host column of 33 chars, fields "LS NABWV")
def mtr_header(host):
    return ("HOST: %-33s" % host)[:33] + " Loss%   Snt   Last   Avg  Best  Wrst StDev"
def mtr_row(i, name, loss, snt, last, avg, best, wrst, stdev):
    return (" %2d.|-- %-33s" % (i, name))[:33] + " %4.1f%%" % loss + " %5d" % snt + " " + \
        "".join(" %5.1f" % v for v in (last, avg, best, wrst, stdev))
mtr_text = """\
HOST: trainer-a                   Loss%   Snt   Last   Avg  Best  Wrst StDev
  1.|-- 10.10.0.1                  0.0%    10    0.3   0.3   0.2   0.5   0.1
  2.|-- 172.31.255.2               0.0%    10   11.9  12.0  11.8  12.4   0.2
  3.|-- 10.20.0.15                 0.0%    10   12.2  12.2  12.0  12.5   0.1"""
assert mtr_text.splitlines() == [
    mtr_header("trainer-a"),
    mtr_row(1, "10.10.0.1", 0.0, 10, 0.3, 0.3, 0.2, 0.5, 0.1),
    mtr_row(2, "172.31.255.2", 0.0, 10, 11.9, 12.0, 11.8, 12.4, 0.2),
    mtr_row(3, "10.20.0.15", 0.0, 10, 12.2, 12.2, 12.0, 12.5, 0.1),
]

# router capture: ICMP 0.277 ms after the oversized segment; 576-byte IP packet, 556 bytes of ICMP
assert round((40.508390 - 40.508113) * 1000, 3) == 0.277
assert 576 - IP == 556 and 556 - ICMP == 548 and 3851143789 - 3851142341 == 1448

# capture after clamping
after = [parse(l) for l in """\
12:03:17.402211 IP 10.10.0.21.40566 > 10.20.0.15.8080: Flags [S], seq 2210384467, win 64240, options [mss 1360,sackOK,TS val 815120987 ecr 0,nop,wscale 7], length 0
12:03:17.414744 IP 10.20.0.15.8080 > 10.10.0.21.40566: Flags [.], seq 124:1472, ack 112, win 509, options [nop,nop,TS val 2950015444 ecr 815120999], length 1348: HTTP
12:03:17.414756 IP 10.20.0.15.8080 > 10.10.0.21.40566: Flags [.], seq 1472:2820, ack 112, win 509, options [nop,nop,TS val 2950015444 ecr 815120999], length 1348: HTTP
12:03:17.426533 IP 10.10.0.21.40566 > 10.20.0.15.8080: Flags [.], ack 2820, win 501, options [nop,nop,TS val 815121011 ecr 2950015444], length 0""".splitlines()]
assert after[1]["len"] == after[2]["len"] == 1360 - TS_OPT
assert after[3]["ack"] == 124 + 2 * 1348
assert round((after[3]["t"] - after[2]["t"]) * 1000, 1) == 11.8

# tunnel overheads with an outer IPv4 header (WireGuard with an IPv6 outer header gives the 1420 default)
overhead = {"IPIP": 20, "GRE": 20 + 4, "VXLAN": 20 + 8 + 8 + 14, "WireGuard/IPv4": 20 + 8 + 32, "WireGuard/IPv6": 40 + 8 + 32}
assert {k: 1500 - v for k, v in overhead.items()} == {
    "IPIP": 1480, "GRE": 1476, "VXLAN": 1450, "WireGuard/IPv4": 1440, "WireGuard/IPv6": 1420}
print("network-debugging-roleplay-b checks ok")
```

</details>

</details>
