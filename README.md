## Small, specific utilities for performance evaluations

They are not designed to be portable. Linux, x86_64.

### fillone

Block device (or filesystem) load generator and performance evaluator. Like Flexibe I/O Tester (git://git.kernel.dk/fio.git) but the fillone is older and I wrote it.

- Available loads: seq. read, seq. write, rnd. read, rnd. write.
- Parralelism: aio() based.
- Can write a complex pattern that cannot be compressed or deduplicated. (Hence its name: do not fill it with zero, but with something else.)
- As far as I know this is the only tool that could make sequential load with multiple outstanding IO operation. And thus a little faster than `dd` itself and a little faster even any linux IO scheduler.

### memeater

Memory consumption load generator. (It doesn't measure itself.)

- It occupies and fills as much memory as it can.
- You can test your system in verious OOM scenarios. For example hard paging speed or increased DB latency. Or how the oom-kill works.

### fslatency

Filesystem write+fdsync latency measuring tool. And little load generator.

- It measures the write latency ten times every second.
- Write the results in a table. For later processing.
- Can be used for check filesystem performance stability in various scenarios:
  - in case of failover
  - in case of a huge load

### lagmeter (lagmeter.c, meeting.h)

Linux kernel scheduler (and underlying hardware) measuring tool. It's a bit tricky to catch a goodness of the scheduler.
It can be used to compare different settings.
I wrote it when the CFS scheduler will become NUMA-aware in Linux.

- pthread-based
- It starts many threads, and measures how long it takes the kernel to hand over control to a thread.
- Due to the nature of the matter, the measurements are very delicate and sensitive, so the complete result table is not written anywhere.
Writing is orders of magnitude slower than cpu scheduling. Only the average/min/max scheduling delay is displayed.
- It measures one every microsecond.
