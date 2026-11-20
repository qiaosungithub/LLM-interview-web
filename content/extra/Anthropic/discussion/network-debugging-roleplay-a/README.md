# Network Debugging Role Play A: Intermittent Connection Resets on Reused Connections

## Problem

The interviewer plays an on-call colleague who has the logs, the dashboards and access to the machines; you lead the investigation: ask questions, ask for data, have them run commands, and make the calls. Each step's prompt contains what the interviewer revealed after the previous step, so you can practice one step at a time. The scenario, numbers and command output are all made up by us; the output formats were checked against the real tools, but the values are not measurements.

### Opening: narrowing it down

**Q1.** Your colleague says: "The order service runs in a private subnet on AWS and calls a partner's payment API, `https://api.partner.example`, from Python. About 0.3% of calls fail with the error below, and one retry almost always fixes it. The partner says everything is fine on their side." What do you ask first, and what data do you ask for?

```text
requests.exceptions.ConnectionError: ('Connection aborted.', ConnectionResetError(104, 'Connection reset by peer'))
```

**Q2.** Here is what your questions turned up: (1) it started about three weeks ago, when the nightly batch job was moved to a different service; there were no releases around that time; (2) it happens on every machine, and the failures cluster between 01:00 and 06:00 UTC, exactly when traffic is lowest; (3) every failed request errors out within 1 millisecond of being sent, and every one of them used an old connection taken from the connection pool; newly created connections have never failed; (4) the partner can find no record of any of the failed requests, and their load balancer's idle timeout is 1 hour. At your request, your colleague took a week of requests from the client logs and grouped them by "how long since this connection was last used" (called the idle time below); each bucket includes its left end and excludes its right end. They also checked DNS.

| Connection idle time | Requests | Failures |
|---|---:|---:|
| 0–60 s | 1,180,000 | 0 |
| 60–300 s | 76,500 | 0 |
| 300–350 s | 9,300 | 0 |
| 350–600 s | 2,940 | 2,940 |
| 600 s and up | 860 | 860 |

```text
$ dig +noall +answer api.partner.example
api.partner.example.	60	IN	A	203.0.113.50
api.partner.example.	60	IN	A	203.0.113.51
```

What do you read from this data? What hypotheses are there, which fits best, and what do you do next?

### Packet capture: who sent the RST

**Q3.** Your colleague captured a night of traffic on one machine (10.0.12.34): `sudo tcpdump -nn -i ens5 -w partner.pcap 'host 203.0.113.50 and tcp port 443'`, then picked out one failed connection (local port 48316) to show you. It starts with the three-way handshake:

```text
$ TZ=UTC tcpdump -nn -r partner.pcap 'tcp port 48316'
02:07:14.201455 IP 10.0.12.34.48316 > 203.0.113.50.443: Flags [S], seq 1946283124, win 64240, options [mss 1460,sackOK,TS val 1193046212 ecr 0,nop,wscale 7], length 0
02:07:14.239611 IP 203.0.113.50.443 > 10.0.12.34.48316: Flags [S.], seq 3772190654, ack 1946283125, win 65160, options [mss 1460,sackOK,TS val 3320158872 ecr 1193046212,nop,wscale 7], length 0
02:07:14.239652 IP 10.0.12.34.48316 > 203.0.113.50.443: Flags [.], ack 1, win 502, options [nop,nop,TS val 1193046250 ecr 3320158872], length 0
```

The TLS handshake lines in between are omitted; below is everything else on this connection:

```text
02:07:14.363504 IP 10.0.12.34.48316 > 203.0.113.50.443: Flags [P.], seq 598:1121, ack 4467, win 501, options [nop,nop,TS val 1193046374 ecr 3320158958], length 523
02:07:14.401871 IP 203.0.113.50.443 > 10.0.12.34.48316: Flags [P.], seq 4467:5162, ack 1121, win 501, options [nop,nop,TS val 3320159034 ecr 1193046374], length 695
02:07:14.401899 IP 10.0.12.34.48316 > 203.0.113.50.443: Flags [.], ack 5162, win 497, options [nop,nop,TS val 1193046413 ecr 3320159034], length 0
02:14:07.482913 IP 10.0.12.34.48316 > 203.0.113.50.443: Flags [P.], seq 1121:1533, ack 5162, win 497, options [nop,nop,TS val 1193459494 ecr 3320159034], length 412
02:14:07.483329 IP 203.0.113.50.443 > 10.0.12.34.48316: Flags [R], seq 3772195816, win 0, length 0
```

Read it line by line: who sent the RST, roughly where did it come from, and why at this moment?

**Q4.** You ask how traffic leaves this machine. Your colleague says the private subnet's default route (0.0.0.0/0) points to a NAT gateway, and all traffic to the partner goes through it. How do you confirm your diagnosis instead of relying on reasoning alone?

### Explanation and fix

**Q5.** Your confirmation results: on the same machine, sending a request after the connection has been idle for 340 seconds succeeds 20 times out of 20; after 360 seconds idle, it is reset 20 times out of 20. Running the same test from a machine with a public IP that does not go through the NAT gateway works fine even after 900 seconds idle. The NAT gateway's CloudWatch metric `IdleTimeoutCount` is always above 0, with a peak every night. Your colleague says: "Weird, we definitely have TCP keepalive turned on," and shows you:

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

Why didn't keepalive help? The client library checks whether a connection is still alive before taking it from the pool; why didn't that catch it either?

**Q6.** You ask how the client is written. Your colleague says: it uses Python `requests` (urllib3 2.x underneath), one global `Session`, the default `HTTPAdapter`, and no socket options set; the failed calls include both GETs that query order status and POSTs that create payments. You need to stop the bleeding tonight. What do you change, and how do you verify the fix?

**Q7.** After the change ships, the next night the two groups of requests idle for 350 seconds or more no longer fail, and `IdleTimeoutCount` drops to near 0. The interviewer asks: "How do you prevent this in the long run? Does this conclusion still hold on a different cloud or different network gear? If the middlebox, after timing out, silently drops packets instead of replying with an RST, what would the symptoms look like?"

### Retrospective

**Q8.** The interviewer finally asks: "This problem lurked for three weeks. How could you have caught it earlier, or seen it coming when the nightly batch job was moved?"

## Reference solution

<details>
<summary>Show the reference solution</summary>

There is very little evidence about this round: the process page only says that the specialized rounds for performance and infrastructure roles include a "network-debugging role play", and one performance-engineering candidate's candidate portal also listed it; nobody has recorded the scenario, the length or how it is scored. What follows is written according to how debugging role plays generally go. The interviewer is most likely watching for these things: whether you proactively ask the right questions; whether the data you ask for at each step can distinguish between hypotheses; whether you can read raw output (tcpdump, ss, sysctl); whether you stop the bleeding first and then fix the root cause; and whether you can run a retrospective at the end.

How to practice: look only at the prompt, say out loud what you would ask at this step, what data you want and what you expect to see, then expand the answer and compare. The last paragraph of each answer is "The interviewer reveals", which is the information at the start of the next question. Common ways to lose points: jumping in without asking; blaming the other side or "the network" right away; not reading raw output line by line; only adding retries, which hides the problem; not verifying after the fix.

### Opening: narrowing it down

**Q1.** A good approach is to first ask questions that cut the search space in half, each tied to a hypothesis:

- When did it start? What changed around then: releases, configuration, infrastructure, and also traffic patterns?
- Does it happen on all machines or only a few? Only with this partner, or with every external API?
- Is there a time pattern: peaks, troughs, top of the hour?
- What does a failure look like: how long after the request is sent does the error come back? An immediate error and a long wait followed by a timeout are two completely different classes of problems.
- Do failures happen on newly created connections, or on old connections reused from the pool? Ask your colleague to add "how long since this connection was last used" to the logs and compute the failure rate grouped by it.
- How does traffic leave: directly, through NAT, a proxy or a firewall?
- Can the partner find the failed requests on their side? What is their load balancer's idle timeout?

You can mention in passing: "Connection reset by peer" only means this host received an RST (reset) packet; it does not mean the remote server sent it. A device on the path can also send an RST spoofing the remote address.

Ways to lose points: declaring "it's the partner's problem" straight away; asking for packet captures on every machine before narrowing down; turning up the retry count first and hiding the problem.

The interviewer reveals: the four facts at the start of Q2, plus the statistics grouped by idle time and the dig result.

**Q2.** Read the table first: of the 1,265,800 requests idle for less than 350 seconds, not one failed; all 3,800 requests idle for 350 seconds or more failed. In total, 1,269,600 requests and 3,800 failures, a failure rate of 0.30%, exactly the reported number. A cutoff this clean means something "expires after about 350 seconds of idleness".

Then go through the hypotheses one by one:

- The partner server's keep-alive timeout: when a server closes an idle connection itself, it sends a FIN, a normal close; the client library notices the connection is closed before reusing it and never sends the request. Also, common defaults are not 350 seconds (for example nginx's `keepalive_timeout` defaults to 75 seconds), the partner's load balancer is at 1 hour, and the partner can't find these requests. Doesn't fit.
- A device on our side of the path (NAT, firewall, load balancer) has an idle timeout around 350 seconds: the device deletes the state for this connection, and when the next packet arrives it replies with an RST. Fits: the cutoff is clean; requests fail as soon as they are sent; the partner receives nothing; at night traffic is low, so connections easily sit idle for a long time.
- DNS changed, and old connections are still attached to an IP that was taken out of service: dig shows both IPs present with a 60-second TTL; and this kind of problem would not split neatly by idle time. Doesn't fit.
- The other side is overloaded or rate limiting: that should show up at peak hours, and new connections would fail too. Doesn't fit.

Why it only started three weeks ago: the nightly batch job used to keep these connections busy, so no connection could sit idle for 350 seconds; after the batch job moved away, connections at night often sit idle for several minutes.

Next step: capture packets on one machine and see where the RST comes from, when it arrives, and whether there are any packets during the idle period.

Ways to lose points: still discussing DNS after seeing this table; not connecting "three weeks ago" with "the batch job moved away".

The interviewer reveals: the capture of one failed connection (Q3).

### Packet capture: who sent the RST

**Q3.** Line by line:

- The first three lines are the three-way handshake. From sending the SYN to receiving the SYN-ACK took 38.156 ms; that is the round-trip time (RTT) to the partner, about 38 ms.
- The first three lines of the second part are the previous normal use of this connection: send a request, receive a response, client ACK. The relative sequence numbers start at 598 because the omitted TLS handshake already used up the earlier bytes.
- At 02:14:07.482913, after 413.081 seconds idle, the client sends a new request. There were no packets either way during the idle period: no FIN and no keepalive probes. The timestamp option (TS val) agrees: the client's TS val went from 1193046413 to 1193459494, a difference of 413081; Linux's TCP timestamp clock ticks once per millisecond, which is exactly 413.081 seconds.
- In the last line, the RST arrives 0.416 ms after the request was sent. The partner is 38 ms away, so its reply cannot come that fast; this RST can only come from a device very close to this host, spoofing the partner's address.
- The RST shows its sequence number as the absolute value 3772195816 (tcpdump does not convert packets without the ACK flag to relative sequence numbers). The server's initial sequence number is 3772190654; adding 5162 gives exactly 3772195816, the point the client had already acknowledged (`ack 5162`). RFC 9293 says that if the connection does not exist and an incoming segment carries an ACK, the reply is `<SEQ=SEG.ACK><CTL=RST>`, i.e. an RST that uses the incoming segment's acknowledgment number as its sequence number. This is exactly that shape: whoever sent the RST no longer has this connection.

Conclusion: after 413 seconds idle, a device on the path very close to this host had already forgotten the connection, and as soon as the client sent data, it replied with an RST. The next question: which devices does traffic from this machine pass through?

Ways to lose points: seeing "there's an RST" and concluding the partner sent it; ignoring the timing; not checking whether there were packets during the idle period.

The interviewer reveals: the subnet's default route points to a NAT gateway (Q4).

**Q4.** Three kinds of evidence; ideally get all of them:

- Check the docs: AWS's NAT gateway troubleshooting docs say that a connection through a NAT gateway times out after it has been idle for 350 seconds or more; after the timeout, if the resource behind it sends more packets on that connection, the NAT gateway replies with an RST (not a FIN); the fix is to keep traffic flowing on the connection, or to enable TCP keepalive on the instance with an interval under 350 seconds.
- Look at the metrics: the NAT gateway's CloudWatch metric `IdleTimeoutCount` is, per the docs, the number of connections that went from active to idle because they were not closed gracefully and had no activity for 350 seconds; the docs say an increase may mean clients behind the NAT gateway are reusing stale connections. Check whether it moves in step with the failures and whether it peaks at night.
- Run a controlled experiment: on the same machine, open a connection, leave it idle for 340 seconds and for 360 seconds before sending a request, 20 times each; then run the same test from a machine with a public IP that doesn't go through the NAT gateway, as a control. Use a GET endpoint with no side effects for the requests.

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

Prediction: through the NAT gateway, every 340-second run succeeds and every 360-second run fails; on the control machine, all succeed.

Ways to lose points: announcing the root cause based on reasoning alone; no control; creating a new connection every time in the experiment, which can never hit the idle timeout.

The interviewer reveals: the experiment and metric results (start of Q5), and the colleague saying "we have keepalive on".

### Explanation and fix

**Q5.** Three layers of explanation:

1. These three sysctls are only defaults; they apply only to sockets that set the `SO_KEEPALIVE` option (see the tcp(7) man page). `ss -o` shows `timer:(keepalive,…)` for connections with keepalive enabled; neither connection here has it, so it was never turned on. urllib3's default socket options contain only `TCP_NODELAY`, not `SO_KEEPALIVE`.
2. Even if it were on, with these defaults a connection has to be idle for 7200 seconds before the first probe goes out, far more than 350 seconds, so it could never keep up. With the defaults, declaring the peer dead takes $7200 + 9 \times 75 = 7875$ seconds.
3. When urllib3 takes a connection from the pool, it calls `is_connection_dropped`, which does a non-blocking "is the socket readable right now" check: if the peer sent a FIN or RST, the socket becomes readable and the check catches it. But when the NAT gateway times out, it sends nothing; the socket looks perfectly healthy, and you only get the RST once you send data. So the check can't catch it.

You can add: receiving this kind of RST increments the kernel counter `TcpEstabResets` (visible with `nstat -az TcpEstabResets`), which you can use for monitoring later.

Ways to lose points: seeing the sysctls and assuming keepalive is on; not knowing the default is 7200 seconds; thinking the pool's check can detect a middlebox timeout.

The interviewer reveals: the client is Python requests, with a global Session, the default HTTPAdapter, and both GETs and POSTs (start of Q6).

**Q6.** Tonight's change: mount an adapter with keepalive enabled on this Session, so idle connections send a probe every 120 seconds (under 350 seconds, with plenty of margin). The probes pass through the NAT gateway and reset its idle timer.

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

`TCP_KEEPIDLE`, `TCP_KEEPINTVL` and `TCP_KEEPCNT` are the Linux option names; on macOS and other systems the names and support differ somewhat.

Retries need handling too:

- Only automatically retry idempotent requests, such as the GET that queries order status. POST is already absent from the methods urllib3's `Retry` allows to be retried by default.
- The POST that creates a payment must not be retried blindly, or the customer might be charged twice. Either use an idempotency key, if the partner supports one, or check the status first and then decide. In this scenario the RST happens before the request reaches the partner, but the client generally can't be sure of that, so you can't take it as a given.

Verification:

- Run `ss -tno state established dst 203.0.113.50` on the machine; every connection should carry `timer:(keepalive,…)` with no more than two minutes remaining;
- The next day, look at the failure rate grouped by idle time; the two groups at 350 seconds and above should drop to 0;
- `IdleTimeoutCount` drops to near 0.

Another option is to have the client discard connections that have been idle too long (for example, close and reconnect anything idle for more than 300 seconds). It works just as well, but requests and urllib3 have no built-in parameter for it, so you'd have to write a wrapper. Also, keepalive makes connections live longer; a NAT gateway supports at most 55,000 concurrent connections per IPv4 address to each unique destination (same destination IP, port and protocol), so don't make the connection pool too large.

Ways to lose points: only adding retries without keepalive; automatically retrying POSTs too; not verifying after the change.

The interviewer reveals: the results after the rollout (start of Q7).

**Q7.** The long-term principle: every layer a connection passes through has its own idle timeout. The client pool's idle limit has to be smaller than the server's keep-alive timeout, and also smaller than the idle timeout of every middlebox on the path; if you can't guarantee the latter, make the keepalive interval shorter than the shortest idle timeout on the path. Concretely:

- Put these settings into the company's shared HTTP client library instead of relying on every service to remember them;
- Document, for each environment, which devices are on the path and what their idle timeouts are;
- Clients must set a read timeout; never wait forever.

You can't just reuse 350 seconds everywhere; the number varies by vendor and configuration. Below are the defaults found in the official docs when this question was written:

| Device | Idle timeout | After the timeout |
|---|---|---|
| AWS NAT gateway | 350 s; no option to change it was found in the docs | Replies with an RST when the next packet arrives |
| AWS Network Load Balancer (NLB), TCP listener | 350 s by default, configurable from 60 to 6000 s | Replies with an RST when the next packet arrives |
| Azure Load Balancer | 4 minutes by default; load-balancing rules and inbound NAT rules can be set to 4–100 minutes, outbound rules to 4–120 minutes | Silently drops by default; a "TCP reset" option can be enabled |
| Google Cloud NAT | 1200 s for established TCP connections by default, adjustable | — |
| Linux conntrack (self-managed NAT or firewall) | `nf_conntrack_tcp_timeout_established` defaults to 432000 s (5 days) | Depends on how the firewall rules handle it |
| RFC 5382 requirement for NATs | The idle timeout for established connections must not be shorter than 2 hours 4 minutes | — |

Defaults also differ by language: Go's `http.DefaultTransport` enables TCP keepalive at 30 seconds by default and closes idle connections after 90 seconds, so the same path would not hit this problem; Python requests has neither.

The silent-drop variant (for example the Azure Load Balancer's default behavior) is harder to debug: the client doesn't error out immediately; it hangs. The request goes out with no response, and the kernel retransmits again and again with exponential backoff, only giving up once `tcp_retries2` (default 15) is exhausted; the kernel docs say this corresponds to a theoretical timeout of 924.6 seconds, and that is only a lower bound. Without a read timeout, what the application sees is a request hanging for more than ten minutes. Countermeasures: always set a read timeout on the client; when needed, use `TCP_USER_TIMEOUT` to cap how long sent data may go unacknowledged; and still enable keepalive.

Ways to lose points: treating 350 seconds as a universal rule; fixing only this one service; not considering the silent-drop case.

The interviewer reveals: the problem is solved; moving on to the retrospective (Q8).

### Retrospective

**Q8.** Cover monitoring, testing and the change process:

- Monitoring: failure rate grouped by connection idle time (that table was the key evidence this time); an alert on the NAT gateway's `IdleTimeoutCount`; growth of `TcpEstabResets` on the hosts; break statistics out by error type, and don't lump connection resets in with ordinary 5xx errors.
- Active probing: a scheduled job that deliberately leaves a connection idle for, say, 400 seconds before sending a request, so you notice as soon as an idle timeout changes.
- Testing: add a "stop traffic for a few minutes, then resume" phase to load tests or soak tests; testing only under sustained high traffic will never catch an idle timeout.
- Logging: the client records for each request whether it used a new or reused connection, and how long the connection was idle before reuse.
- Change process: a change in traffic pattern (such as moving the nightly batch job away) is also a change; check whether it will leave connections idle for longer; every time you add a layer of network gear, record its idle timeout.
- Don't let retries mask problems: errors that succeed on one retry are the easiest to ignore; track the fraction of requests that "only succeeded thanks to a retry" separately and set a threshold on it.

Ways to lose points: only saying "add monitoring" without saying what to monitor; only describing the fix for this one service, not the process.

<details>
<summary>Verification code (runnable)</summary>

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
