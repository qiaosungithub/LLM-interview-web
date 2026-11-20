# Network Debugging Role Play B: Small Files Download, Large Files Hang

## Problem

The interviewer plays an on-call colleague who has the logs, the dashboards and access to the machines; you lead the investigation: ask questions, request data, have them run commands, and make the calls. Each step's question includes what the interviewer revealed in the previous step, so you can practice it step by step. The scenario, numbers and command output are all made up by us; the output formats were checked against the real tools, but the values are not measured.

### Opening: narrowing it down

**Q1.** Your colleague says: "trainer-a (10.10.0.21), a training machine in the DC1 data center, has to download model files from the artifact server `artifacts.dc2.internal` (10.20.0.15, port 8080) in the DC2 data center, and for the past few days it keeps hanging and eventually timing out. But the service's health check has been green the whole time." They ran two commands on trainer-a:

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

What do you read from these two outputs? What do you ask next?

**Q2.** What you found out: (1) last Thursday, the dedicated line between the two data centers was replaced by a tunnel through the routers; the MTU of the tunnel interface `tun0` on both routers, rtr-dc1 and rtr-dc2, is 1400, the servers' NIC MTUs are all 1500, and nothing else was adjusted; (2) since then, every cross-DC download of a large file hangs, while downloading the same file inside DC2 works fine; (3) uploading a large file from trainer-a to DC2 also hangs, and small cross-DC requests all work. Your colleague also ran ping and mtr on trainer-a:

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

Your colleague says: "No packet loss, latency is normal, so it's probably not the network, right?" What do you think? What are your hypotheses, and what's the next step?

### Localizing: where the packets get stuck

**Q3.** As you asked, your colleague ran two sets of pings with fragmentation disallowed (the DF bit) on trainer-a:

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

What do these two results tell you? Is there anything else worth noticing in the output of the first set?

**Q4.** Your colleague starts a capture on the server 10.20.0.15: `sudo tcpdump -nn -i eth0 -w dl.pcap '(host 10.10.0.21 and tcp port 8080) or icmp'`, then runs the download on trainer-a again and stops the capture about 15 seconds later. Here is everything captured:

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

Before stopping the capture, your colleague looked at `ss -tin` for this connection on the server (only the relevant fields are kept below; the rest is elided with `...`) and at the route cache:

```text
	 cubic wscale:7,7 rto:13568 backoff:6 ... mss:1448 pmtu:1500 ... advmss:1448 cwnd:1 ... unacked:12 retrans:1/6 lost:12 ...
$ ip route get 10.10.0.21
10.10.0.21 via 10.20.0.1 dev eth0 src 10.20.0.15 uid 1000 
    cache 
```

Read it line by line: which packets arrived, which didn't, and what is the server doing? What kind of mechanism is the problem in? Where do you look next?

### Mitigation and root fix

**Q5.** Your colleague captures on rtr-dc2's interface facing the servers while trainer-a starts another download. Here are two lines from it:

```text
$ TZ=UTC tcpdump -nn -S -r rtr-dc2-lan.pcap
11:26:40.508113 IP 10.20.0.15.8080 > 10.10.0.21.40388: Flags [.], seq 3851142341:3851143789, ack 1405993833, win 509, options [nop,nop,TS val 2948891572 ecr 813997132], length 1448: HTTP
11:26:40.508390 IP 10.20.0.1 > 10.20.0.15: ICMP 10.10.0.21 unreachable - need to frag (mtu 1400), length 556
```

They also found: between rtr-dc2 and the server subnet there is a firewall in transparent (bridge) mode that only allows ICMP echo request and echo reply (used by ping) and time exceeded (used by traceroute), and drops all other ICMP; the DC1 side uses the same rule template. Changing the firewall rules has to go through review, tomorrow at the earliest; on-call can change the router configuration directly and get it live in ten minutes. What is the root cause? How do you mitigate it right now? What are the options, what does each cost, and which do you pick?

**Q6.** An hour later, MSS clamping has been added on both routers and cross-DC downloads work again. Your colleague captured another new connection on the server; here is the beginning (the remaining data segments are omitted):

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

Read these lines. Why is the mss different in the two directions? What is the long-term root fix? Which cases does MSS clamping not cover? Once the firewall is changed, how do you verify it?

### Retrospective

**Q7.** The interviewer's last question: "After the tunnel went live, it took several days before anyone reported the problem, and the health check was green the whole time. How could we have caught it sooner, or avoided it when switching to the tunnel?"

## Reference solution

<details>
<summary>Show the reference solution</summary>

There is very little evidence about this kind of round: the process page only says that the specialized rounds for performance and infrastructure roles include a "network-debugging role play", and that one performance engineering candidate's candidate portal also listed it; nobody has recorded the scenario, the length or the grading. What follows is written according to how debugging role plays generally go. The interviewer most likely watches for these things: whether you proactively ask the right questions; whether the data you ask for at each step can distinguish between hypotheses; whether you can read raw output (curl, ping, tcpdump, ss); whether you mitigate first and then fix the root cause, and spell out the cost of each approach; and whether you can run a retrospective at the end.

How to practice: look only at the question, first say out loud what you would ask at this step, what data you would want and what you expect to see, then expand the answer and compare. The last paragraph of each answer, "Interviewer reveals", is the information at the start of the next question. Common ways to lose points: ruling out the network because ping works; not reading raw output line by line; not asking what changed recently; not being able to name the cost of a mitigation; not verifying after the fix.

### Opening: narrowing it down

**Q1.** First read the curl output line by line; it has already ruled out a lot:

- DNS resolution works, the TCP connection was established (`Connected to`), the request went out, and the server's response headers arrived: `200 OK`, `Content-Length: 5242880` (5 MiB).
- But not a single byte of the response body arrived (`0 out of 5242880 bytes received`), and it timed out after 20 seconds. It isn't "slow"; it's "not moving at all".
- The `/healthz` response is only two bytes, and the whole request completed quickly.

So the service itself is alive and the connection works. Small amounts of data get through, but the moment it starts sending large chunks of data it hangs, which looks a lot like "large packets can't get through". A slow server or slow disk usually transfers slowly; it doesn't stall at exactly zero bytes.

What to ask next:

- When did it start? Did anything change in the network around then: a new line, a tunnel, a firewall, an MTU change?
- Is it just this machine, or every machine across the two DCs? Does downloading the same file inside the data center work?
- Does the reverse direction (sending a large file from trainer-a to DC2) work?
- Do small files work, and at roughly what size does it stop working?
- Where is the health check run from, and how much data does it transfer?

Ways to lose points: concluding the server is fine because the health check is green and going off to investigate the client; not reading the byte count in the curl output; restarting the service right away.

Interviewer reveals: the situation at the start of Q2, and the ping and mtr results.

**Q2.** Disagree with "it's not the network". ping and mtr both use small packets: by default ping sends 56 bytes of data, 84 bytes in total with the ICMP and IP headers; mtr's default probe is 64 bytes. They only show that small packets get through; they say nothing about whether a 1500-byte packet can.

What you can read from the output: ttl=62 means the reply went through two routers (Linux's default initial TTL is 64, minus 2); mtr's three hops are rtr-dc1's internal address, rtr-dc2's tunnel address and the server itself, consistent with the tunnel topology; the round-trip time is about 12 milliseconds, with no loss.

Hypotheses:

1. A path MTU problem: the tunnel interface's MTU is only 1400, and the server sends 1500-byte packets with the DF bit set (no fragmentation allowed), which can't enter the tunnel. Normally the router sends back an ICMP "fragmentation needed" message telling the sender to make its packets smaller; if something drops that message, the sender never finds out and keeps resending packets of the same size. This is called a PMTUD black hole. It fits every symptom: started after the tunnel switch last Thursday, fine inside the data center, both directions hang, small requests work, ping works.
2. A device on the path (a firewall, an intrusion prevention system) drops packets based on content or size. Possible, but the timing lines up with the tunnel going live, so check the first one first.
3. A window or buffer configuration problem: that would make it very slow, not deliver zero bytes. Doesn't fit.
4. A server or disk problem: downloads inside the data center work. Doesn't fit.
5. Packet loss in the tunnel: ping and mtr show no loss, and loss would make transfers slow, not stall at the same point every time. Doesn't fit.

Next step: test large packets with DF-bit pings and binary-search the path MTU. `-s` is the number of bytes of ICMP data; add the 8-byte ICMP header and the 20-byte IP header to get the IP packet size: `-s 1472` corresponds to 1500 bytes, `-s 1372` to 1400 bytes. `-M do` sets the DF bit and also stops the local host from fragmenting.

Ways to lose points: agreeing that "it's not the network"; not knowing that ping and mtr use small packets by default; not connecting "the tunnel switch last Thursday" with the symptoms.

Interviewer reveals: the results of the two sets of DF pings (Q3).

### Localizing: where the packets get stuck

**Q3.** All 1500-byte packets are lost and all 1400-byte packets get through, so the path MTU is between 1400 and 1499, matching the tunnel interface's MTU of 1400. For the exact value you can keep binary-searching, e.g. try `-s 1373` (1401 bytes). The successful replies show `1380 bytes`, which is the size of the ICMP echo reply (1372 bytes of data plus the 8-byte ICMP header), not including the IP header.

What is more noteworthy in the first set's output is that "nothing was printed". If the ICMP "fragmentation needed" message sent by rtr-dc1 had reached trainer-a, ping would print an error line like `From 10.10.0.1 icmp_seq=1 Frag needed and DF set (mtu = 1400)`, and the kernel would remember the MTU for this path, so later packets would fail locally. There is nothing at all, which means that ICMP message was either never sent or dropped along the way. That is exactly the signature of a black hole, and it also explains why the upload direction hangs.

Next step: go back to the reported problem, i.e. the download direction. Capture on the server 10.20.0.15, and the filter must include `icmp`, otherwise "no ICMP seen" means nothing; at the same time look at `ss -tin` for this connection (the MSS, path MTU and retransmission state as the kernel sees them) and `ip route get 10.10.0.21` (whether the kernel has cached a smaller MTU for this destination).

Ways to lose points: only saying "it's an MTU problem" without noticing the key piece of information that there is no ICMP error; capturing only the TCP port and filtering out ICMP.

Interviewer reveals: the capture, `ss` and route cache on the server (Q4).

**Q4.** Read it section by section:

- The first three lines are the three-way handshake. Both SYNs carry `mss 1460`, i.e. both sides believe the path MTU is 1500. The client's ACK took 11.798 milliseconds, which is the round-trip time.
- Lines 4 and 5: the request is 111 bytes; adding the 20-byte TCP header, the 12-byte timestamp option and the 20-byte IP header gives a 163-byte IP packet, which arrived, and the server acknowledged it.
- Line 6: the response headers are 123 bytes, a 175-byte IP packet. The client replied with `ack 124` at 10:41:03.144421, so it arrived too. That is why curl could print the response headers.
- Lines 7–15: 9 full data segments of 1448 bytes each; with the 52 bytes of headers that is exactly 1500 bytes. Together with the response-header packet that is 10 packets, which is the initial congestion window.
- After receiving `ack 124`, the server sent 2 more new segments (in slow start, the window grows by one for each packet acknowledged).
- At 10:41:03.177842 it sent one more new segment: this is a tail loss probe (TLP); when no acknowledgment arrives within about two round-trip times, it sends one more segment as a probe. The exact wait depends on the kernel version and timer granularity.
- From 10:41:03.389842 onward, it keeps retransmitting the first data segment `124:1572`: the first time is 212 milliseconds after the TLP, which is the retransmission timeout (RTO). Linux's RTO is the smoothed round-trip time plus a variance term, and the variance term has a minimum of 200 milliseconds, so here it is about 212 milliseconds; after that the gaps are 0.424, 0.848, 1.696, 3.392 and 6.784 seconds, doubling each time, which is exponential backoff. Every retransmission is still 1448 bytes; the server never makes the packet smaller.
- The client's acknowledgment number stays at 124: not a single 1500-byte packet arrived. The capture filter includes `icmp`, yet there is not a single ICMP message.

The fields in `ss` match: `backoff:6` means it has already backed off 6 times, and `rto:13568` is exactly 212 × 64 milliseconds; `unacked:12` is 12 unacknowledged segments (9 + 2 + 1), `lost:12` is all of them marked lost after the timeout, `retrans:1/6` is 1 retransmission currently in flight and 6 retransmissions in total, and `cwnd:1` is the congestion window dropping to 1 after the timeout. Most important are `pmtu:1500` and the absence of `mtu` in the route cache: the kernel has never received a "fragmentation needed" message.

Conclusion: packets larger than 1400 bytes are being dropped on the way, and the ICMP "fragmentation needed" message (type 3, code 4) that the router should have sent back to the server never reached the server. This is a PMTUD black hole:

- Path MTU Discovery (PMTUD, RFC 1191) works like this: the sender sets the DF bit on its packets; when a router finds a packet too large and not allowed to be fragmented, it drops the packet and sends back an ICMP message carrying the next hop's MTU; on receiving it, the sender makes its packets smaller. Linux TCP does this by default (`net.ipv4.ip_no_pmtu_disc = 0`).
- As soon as that ICMP message is filtered out, the sender keeps resending at the original size, and the connection stalls until it runs out of retransmissions. RFC 2923 describes this problem specifically.

One more detail: tcpdump on the server sees packets before they enter the protocol stack, so even if the local iptables dropped ICMP, tcpdump would still see it. Here even tcpdump saw nothing, which means the ICMP was gone before it reached the server's NIC, or was never sent at all. Next step: capture on rtr-dc2 to see whether it sends the ICMP; and at the same time ask what other devices sit between rtr-dc2 and the server.

Ways to lose points: seeing retransmissions and saying "heavy packet loss"; not noticing that all the small packets arrived and only the full-size ones didn't; not noticing there is no ICMP in the capture; assuming the server's own firewall dropped the ICMP (tcpdump would see that case).

Interviewer reveals: the capture on rtr-dc2, plus the firewall's rules and the constraints on changing them (Q5).

### Mitigation and root fix

**Q5.** Root cause: rtr-dc2 did its job and sent back the "fragmentation needed (mtu 1400)" ICMP message 0.277 milliseconds later (`length 556` is the length of the ICMP part; the whole IP packet is 576 bytes and quotes the beginning of the original packet), but the transparent firewall in between only allows the ICMP used by ping and traceroute, so it dropped it. DC1 uses the same rules, so the upload direction is the same. ping and mtr work because the ICMP they use happens to be on the allow list, and their packets are all small.

Options you can apply right away:

1. MSS clamping on both routers: when forwarding TCP SYN and SYN-ACK packets, rewrite their MSS option to a smaller value, so both ends only send packets that fit in the tunnel from the start.

   ```text
   iptables -t mangle -A FORWARD -o tun0 -p tcp --tcp-flags SYN,RST SYN -j TCPMSS --clamp-mss-to-pmtu
   ```

   `--tcp-flags SYN,RST SYN` matches packets with the SYN bit set and the RST bit clear, which covers both SYN and SYN-ACK; `--clamp-mss-to-pmtu` computes the path MTU of the outgoing interface minus 40 (IPv4), which here is 1360. You can also hard-code `--set-mss 1360`. It has to be added on both sides: the SYN sent by A determines how large B's packets to A are, and the SYN-ACK sent by B determines how large A's packets to B are. The symptom described in the TCPMSS section of iptables-extensions(8) is exactly this one: the connection is established and then hangs, receiving no data at all. Cost: TCP only; only takes effect for new connections; it papers over the problem, and the firewall's mistake is still there.
2. Turn on `net.ipv4.tcp_mtu_probing = 1` on the servers (the kernel default is 0; some distributions or cloud images change it, so go by the actual value). When set to 1, the kernel enables packetization-layer MTU probing (RFC 4821) only when it suspects a black hole. According to the Linux source, the condition is that the time elapsed since the first retransmission exceeds the model time corresponding to `tcp_retries1` (default 3), i.e. 3 seconds. In this connection's RTO sequence, that is only satisfied at the moment of the 5th retransmission (about 6.6 seconds after the first data segment). After that the MSS drops to around `tcp_base_mss` (the default in recent kernel source is 1024, and reading it on a machine with a 5.19 kernel also gave 1024; the tcp(7) man page says 512, so go by the value you actually read), and then slowly probes upward. Cost: every new connection stalls for a few seconds at the start; it only takes effect on machines where it's turned on; TCP only.
3. Set an MTU per route on the servers, e.g. on the DC2 side `ip route add 10.10.0.0/16 via 10.20.0.1 dev eth0 mtu 1400`, and the reverse route on the DC1 side. Neither cross-DC TCP nor UDP will exceed 1400. Don't add `lock`: according to ip-route(8), with it the kernel stops doing PMTUD and no longer sets the DF bit on IPv4 packets. Cost: every machine has to be changed, and new machines are easy to miss.
4. Change the NIC MTU on every server to 1400: this affects all traffic inside the data center, is the biggest change, and is not recommended.

Recommendation: do MSS clamping on both routers right now — live in ten minutes, with the widest coverage; tomorrow, change the firewall rules through review, which is the real fix. You can also turn on `tcp_mtu_probing = 1` on the servers as a fallback.

Ways to lose points: only saying "lower the MTU" without saying where or what it affects; not knowing that MSS clamping only works for new TCP connections; clamping on only one router; treating the mitigation as the fix and never changing the firewall.

Interviewer reveals: the capture after clamping went live (Q6).

**Q6.** Reading the capture:

- The SYN sent by trainer-a originally said `mss 1460` (its NIC MTU is 1500); passing through rtr-dc1 it was rewritten to 1400 − 40 = 1360, so the server received 1360.
- The server's SYN-ACK says `mss 1460`: the capture point is on the server, before the packet reaches rtr-dc2; rtr-dc2 will rewrite it to 1360, and trainer-a sees 1360.
- The server's subsequent data segments are 1348 bytes each, which is 1360 minus the 12-byte timestamp option; with 52 bytes of headers the IP packet is exactly 1400 bytes, which fits in the tunnel. 11.8 milliseconds later the client acknowledges up to 2820, i.e. both segments arrived.

Root fix:

- Change the firewall: allow ICMP type 3 (destination unreachable), at minimum code 4 (fragmentation needed). For IPv6 it is even more important to allow ICMPv6 type 2 (Packet Too Big): IPv6 routers don't fragment, PMTUD relies entirely on this message, and the minimum link MTU is 1280; for which ICMPv6 messages should not be filtered, see RFC 4890. Change the rule template for both data centers together.
- Keep the MSS clamping as a second line of defense.
- Document the tunnel's MTU and encapsulation overhead. For common encapsulations with an outer IPv4 header and an underlying MTU of 1500:

| Encapsulation | Overhead (bytes) | Tunnel MTU |
|---|---:|---:|
| IPIP | 20 | 1480 |
| GRE (basic header) | 24 | 1476 |
| VXLAN | 50 | 1450 |
| WireGuard (outer IPv4) | 60 | 1440 |
| WireGuard (outer IPv6) | 80 | 1420 (WireGuard interface default MTU) |
| IPsec ESP | Varies with the cipher, the mode and whether NAT traversal is used | Compute from the actual configuration |

Cases MSS clamping doesn't cover: UDP traffic (e.g. QUIC, large DNS responses, home-grown UDP protocols); connections established before the clamping went live, whose MSS is still 1460 and which only recover after reconnecting; and if there is an even smaller MTU further along the path, the value the router computes from its own outgoing interface is wrong too.

Verification after the firewall change:

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

The first packet received rtr-dc1's "fragmentation needed" message, the kernel recorded the MTU for this path as 1400, and the next two packets failed locally; the route cache now has `mtu 1400`, which expires after 600 seconds (`net.ipv4.route.mtu_expires` defaults to 600) and is then rediscovered. Run the same check on the server. You can also use `tracepath` to see the path MTU at each hop. ping itself isn't affected by MSS clamping, so this step proves that PMTUD itself works again. Once ICMP can get through, TCP shrinks its MSS and retransmits immediately on receiving the message, without waiting several seconds for a timeout.

Ways to lose points: only allowing IPv4 ICMP and forgetting ICMPv6; thinking the firewall doesn't need changing once clamping is in place; not knowing which traffic clamping doesn't cover; not verifying PMTUD itself after the change.

Interviewer reveals: the problem is solved; move on to the retrospective (Q7).

### Retrospective

**Q7.** Cover three areas: the change process, monitoring and rules:

- Change process: before and after any change that alters the MTU (adding a tunnel, changing encapsulation, adding a VPN), run DF-bit large-packet pings and `tracepath` in both directions, and actually transfer a file of a few MB. Put "path MTU and MSS clamping" on the network change checklist.
- Health checks should look like real traffic: a `/healthz` that returns only two bytes will never catch a large-packet problem. Add a cross-DC synthetic probe that periodically downloads a file of a few MB and checks speed and integrity.
- Monitor kernel counters (collected with `nstat`): a sudden rise in `TcpExtTCPTimeouts` (retransmission timeouts) and `TcpRetransSegs` (retransmitted segments) on the servers; a rise in `IcmpOutDestUnreachs` and `IcmpMsgOutType3` (destination unreachable messages sent) on the routers with no reaction from the other end is the signal of a black hole; on machines with MTU probing enabled you can also watch `TcpExtTCPMTUPFail` and `TcpExtTCPMTUPSuccess`.
- Firewall rule checks: in rule review or automated checks, forbid rules like "drop all ICMP except ping"; type 3 and ICMPv6 type 2 must be allowed.
- Reporting channel: the hanging downloads went several days before anyone reported them, which means this kind of failure has no alert; counting "request timed out with a 0-byte response body" is a cheap and accurate metric.

Ways to lose points: only saying "add monitoring" without saying what to monitor; only fixing this one incident without changing the change process and the rule template.

<details>
<summary>Checks (runnable)</summary>

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
