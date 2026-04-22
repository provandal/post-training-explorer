#!/usr/bin/env python3
"""
log_templates.py - Multi-line storage error log templates for classification

Each template is a multi-line string (3-5 log lines) that looks like real
syslog / storage-mgr output. Classification emerges from the RELATIONSHIP
between lines, not from any single keyword.

6 categories x 8 templates each = 48 templates total.
Difficulty: 3 EASY + 3 MEDIUM + 2 HARD per category.
"""

import random
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Shared variable pools
# ---------------------------------------------------------------------------
SHARED_VARS = {
    "dev":          ["sda", "sdb", "sdc", "sdd", "sde", "nvme0n1", "nvme1n1", "nvme2n1"],
    "node":         ["stor-node01", "stor-node02", "stor-node03", "stor-node04"],
    "node2":        ["stor-node05", "stor-node06", "stor-node07", "stor-node08"],
    "port":         ["eth0", "eth1", "bond0", "ib0", "fc0", "fc1"],
    "ctrl":         ["A", "B"],
    "pool":         ["pool0", "pool1", "tank0", "data-pool"],
    "vol":          ["vol-prod-01", "vol-db-02", "vol-archive-03", "vol-vdi-04"],
    "count":        [1048576, 2097152, 4194304, 524288, 8388608, 16777216],
    "hours":        [12, 24, 48, 72, 168, 720],
    "mins":         [3, 5, 8, 12, 15, 22, 30, 45],
    "pct":          [78, 82, 85, 88, 91, 93, 95, 97],
    "ms":           [45, 68, 92, 120, 180, 250, 340, 500],
    "baseline_ms":  [8, 12, 15, 18, 22, 25],
    "ver":          ["3.2.1", "3.2.4", "3.3.0", "4.0.1", "4.1.0", "4.1.2"],
    "incident":     ["INC-4821", "INC-5033", "INC-5190", "INC-5402", "INC-5617"],
    "threshold":    [80, 85, 90, 95],
    "ratio":        [0.12, 0.18, 0.25, 0.34, 0.45],
    "rate":         [1.2, 2.5, 3.8, 5.1, 7.4, 12.6],
    "size_tb":      [4.8, 9.6, 14.4, 19.2, 28.8, 48.0],
    "rg":           ["rg0", "rg1", "rg2", "rg3"],
    "host":         ["esxi-host01", "esxi-host02", "k8s-worker03", "db-server04"],
    "site":         ["site-east", "site-west", "site-central"],
}


def generate_timestamps(n: int = 5, base: datetime | None = None,
                        interval_sec: tuple[int, int] = (1, 30)) -> list[str]:
    """Return n ISO-ish syslog timestamps separated by realistic intervals."""
    if base is None:
        base = datetime(2026, 3, 28, 14, 22, 11)
    stamps = []
    t = base
    for _ in range(n):
        stamps.append(t.strftime("%Y-%m-%dT%H:%M:%S"))
        t += timedelta(seconds=random.randint(*interval_sec))
    return stamps


def pick(pool_name: str) -> str:
    """Pick a random value from a shared variable pool."""
    return str(random.choice(SHARED_VARS[pool_name]))


# ---------------------------------------------------------------------------
# TEMPLATES
# ---------------------------------------------------------------------------

TEMPLATES = {
    # ==================================================================
    # DRIVE FAILURE
    # ==================================================================
    "Drive Failure": {
        "templates": [
            # --- EASY 1: clear device I/O errors + SMART failure ---
            "{ts1} {node} kernel: {dev}: I/O error, sector {count}, cmd 0x2a\n"
            "{ts2} {node} smartd: Device /dev/{dev}: SMART Health Status: FAILED\n"
            "{ts3} {node} smartd: Device /dev/{dev}: reallocated sector count = 148, threshold = 5\n"
            "{ts4} {node} kernel: {dev}: media error on read, LBA {count}",

            # --- EASY 2: device timeout + SMART wear-out ---
            "{ts1} {node} kernel: {dev}: cmd 0x28 timeout after 30s\n"
            "{ts2} {node} smartd: Device /dev/{dev}: wear leveling count = 2, min threshold = 10\n"
            "{ts3} {node} kernel: {dev}: device not responding, resetting link\n"
            "{ts4} {node} ctrl-mgr: controller {ctrl} initiated device reseat for /dev/{dev}",

            # --- EASY 3: repeated I/O errors + uncorrectable count ---
            "{ts1} {node} kernel: {dev}: I/O error, sector {count}, cmd 0x2a\n"
            "{ts2} {node} kernel: {dev}: I/O error, sector {count}, cmd 0x28\n"
            "{ts3} {node} smartd: Device /dev/{dev}: uncorrectable error count = 37\n"
            "{ts4} {node} storage-mgr: {vol} marked degraded, rebuild to spare started",

            # --- MEDIUM 1: device errors + latency spike (looks like perf degradation) ---
            "{ts1} {node} storage-mgr: {vol} latency p99={ms}ms (baseline: {baseline_ms}ms)\n"
            "{ts2} {node} kernel: {dev}: I/O error, sector {count}, cmd 0x2a\n"
            "{ts3} {node} smartd: Device /dev/{dev}: current pending sector count = 24\n"
            "{ts4} {node} storage-mgr: {vol} throughput dropped {pct}% correlates with /dev/{dev} errors",

            # --- MEDIUM 2: device errors + network retransmissions (looks like network) ---
            "{ts1} {node} kernel: {dev}: cmd 0x2a timeout after 30s\n"
            "{ts2} {node} network-mgr: {port} retransmissions={rate}/s (elevated)\n"
            "{ts3} {node} smartd: Device /dev/{dev}: SMART Health Status: FAILED\n"
            "{ts4} {node} kernel: {dev}: media error on read, LBA {count}",

            # --- MEDIUM 3: device errors + replication lag (looks like repl issue) ---
            "{ts1} {node} repl-mgr: {vol} replication lag={mins}m to {site}\n"
            "{ts2} {node} kernel: {dev}: I/O error, sector {count}, cmd 0x28, rejected by device\n"
            "{ts3} {node} smartd: Device /dev/{dev}: reallocated sector count = 92, threshold = 5\n"
            "{ts4} {node} storage-mgr: {rg} rebuild started, source: /dev/{dev}",

            # --- HARD 1: device causing controller symptoms (looks like firmware/config) ---
            "{ts1} {node} ctrl-mgr: controller {ctrl} queue depth reached 512, fw={ver}\n"
            "{ts2} {node} ctrl-mgr: controller {ctrl} memory utilization {pct}%\n"
            "{ts3} {node} kernel: {dev}: I/O error, sector {count}, cmd 0x2a\n"
            "{ts4} {node} smartd: Device /dev/{dev}: uncorrectable error count = 64\n"
            "{ts5} {node} ctrl-mgr: controller {ctrl} queue depth returned to normal after /dev/{dev} removed",

            # --- HARD 2: device errors misattributed to firmware (version mentioned but irrelevant) ---
            "{ts1} {node} ctrl-mgr: controller {ctrl} fw={ver} loaded {hours}h ago\n"
            "{ts2} {node} kernel: {dev}: cmd 0x28 timeout after 30s\n"
            "{ts3} {node} smartd: Device /dev/{dev}: wear leveling count = 1, min threshold = 10\n"
            "{ts4} {node} kernel: {dev}: vendor confirmed drive model end-of-life, spare pool exhausted\n"
            "{ts5} {node} storage-mgr: same firmware version running stable on {node2} without /dev/{dev}",
        ],
        "vars": "SHARED_VARS",
    },

    # ==================================================================
    # NETWORK ISSUE
    # ==================================================================
    "Network Issue": {
        "templates": [
            # --- EASY 1: clear link errors + fabric event ---
            "{ts1} {node} network-mgr: {port} link flap detected, down/up in {mins}s\n"
            "{ts2} {node} network-mgr: fabric event: switch port renegotiated at reduced speed\n"
            "{ts3} {node} network-mgr: {port} CRC errors={count} in last {hours}h",

            # --- EASY 2: path failure + retransmissions ---
            "{ts1} {node} network-mgr: {port} path to {node2} marked DOWN\n"
            "{ts2} {node} network-mgr: {port} retransmissions={rate}/s, baseline=0.01/s\n"
            "{ts3} {node} network-mgr: failover to alternate path completed, traffic reverted to {port}",

            # --- EASY 3: multiple link errors + packet loss ---
            "{ts1} {node} network-mgr: {port} RX errors={count} in last {hours}h\n"
            "{ts2} {node} network-mgr: {port} packet loss={ratio}% over {mins}m window\n"
            "{ts3} {node} network-mgr: {port} link speed changed from 100Gbps to 25Gbps\n"
            "{ts4} {node} network-mgr: fabric event: upstream switch reported port errors",

            # --- MEDIUM 1: network causing device timeouts (looks like drive failure) ---
            "{ts1} {node} kernel: {dev}: cmd 0x28 timeout after 30s\n"
            "{ts2} {node} network-mgr: {port} retransmissions={rate}/s, baseline=0.01/s\n"
            "{ts3} {node} network-mgr: {port} path to storage target marked DOWN\n"
            "{ts4} {node} smartd: Device /dev/{dev}: SMART Health Status: OK",

            # --- MEDIUM 2: network causing latency spike (looks like perf degradation) ---
            "{ts1} {node} storage-mgr: {vol} latency p99={ms}ms (baseline: {baseline_ms}ms)\n"
            "{ts2} {node} network-mgr: {port} retransmissions={rate}/s, baseline=0.01/s\n"
            "{ts3} {node} storage-mgr: {vol} throughput dropped {pct}% in last {mins}m\n"
            "{ts4} {node} network-mgr: latency spike correlates with {port} CRC errors={count} in last {hours}h",

            # --- MEDIUM 3: network causing replication failures (looks like repl issue) ---
            "{ts1} {node} repl-mgr: {vol} sync to {site} stalled for {mins}m\n"
            "{ts2} {node} network-mgr: {port} path to {site} marked DOWN, send buffer exhausted\n"
            "{ts3} {node} network-mgr: {port} packet loss={ratio}% over {mins}m window\n"
            "{ts4} {node} repl-mgr: {vol} sync resumed after alternate path activated",

            # --- HARD 1: network issue mentioning firmware + capacity (red herrings) ---
            "{ts1} {node} ctrl-mgr: controller {ctrl} fw={ver} loaded {hours}h ago\n"
            "{ts2} {node} storage-mgr: {pool} utilization={pct}% of {size_tb}TB\n"
            "{ts3} {node} network-mgr: {port} link flap detected, down/up in {mins}s\n"
            "{ts4} {node} network-mgr: {port} CRC errors={count} in last {hours}h\n"
            "{ts5} {node} storage-mgr: {vol} latency normalized after {port} path restored",

            # --- HARD 2: network with controller memory spike (looks like firmware bug) ---
            "{ts1} {node} ctrl-mgr: controller {ctrl} memory utilization {pct}%\n"
            "{ts2} {node} network-mgr: {port} retransmissions={rate}/s causing TCP window collapse\n"
            "{ts3} {node} ctrl-mgr: controller {ctrl} retry queue backed up, depth=4096\n"
            "{ts4} {node} network-mgr: vendor confirmed optic degradation on {port}, path to {node2} marked DOWN\n"
            "{ts5} {node} ctrl-mgr: controller {ctrl} memory utilization returned to baseline after {port} failover",
        ],
        "vars": "SHARED_VARS",
    },

    # ==================================================================
    # CAPACITY WARNING
    # ==================================================================
    "Capacity Warning": {
        "templates": [
            # --- EASY 1: pool utilization + quota alert ---
            "{ts1} {node} storage-mgr: {pool} utilization={pct}% of {size_tb}TB\n"
            "{ts2} {node} storage-mgr: {pool} exceeded threshold at {threshold}%\n"
            "{ts3} {node} storage-mgr: {vol} quota alert: {size_tb}TB of {size_tb}TB used",

            # --- EASY 2: thin provision + snapshot reserve exhaustion ---
            "{ts1} {node} storage-mgr: {pool} thin provision reserve depleted\n"
            "{ts2} {node} storage-mgr: {pool} snapshot reserve usage={pct}%\n"
            "{ts3} {node} storage-mgr: {vol} auto-grow failed, maximum volume size reached\n"
            "{ts4} {node} storage-mgr: {pool} utilization={pct}% of {size_tb}TB",

            # --- EASY 3: pool full + write rejection ---
            "{ts1} {node} storage-mgr: {pool} utilization={pct}% of {size_tb}TB\n"
            "{ts2} {node} storage-mgr: {pool} write operations throttled, free space below minimum\n"
            "{ts3} {node} storage-mgr: {vol} write rejected: pool space exhausted",

            # --- MEDIUM 1: capacity causing write I/O errors (looks like drive failure) ---
            "{ts1} {node} kernel: {dev}: I/O error, sector {count}, cmd 0x2a\n"
            "{ts2} {node} storage-mgr: {pool} utilization={pct}% of {size_tb}TB\n"
            "{ts3} {node} smartd: Device /dev/{dev}: SMART Health Status: OK\n"
            "{ts4} {node} storage-mgr: {pool} write operations throttled, free space below minimum",

            # --- MEDIUM 2: capacity causing latency spikes (looks like perf degradation) ---
            "{ts1} {node} storage-mgr: {vol} latency p99={ms}ms (baseline: {baseline_ms}ms)\n"
            "{ts2} {node} storage-mgr: {pool} utilization={pct}% of {size_tb}TB\n"
            "{ts3} {node} storage-mgr: {host} presenting {count} concurrent I/O requests during rebuild\n"
            "{ts4} {node} storage-mgr: {pool} garbage collection backlog={count} segments",

            # --- MEDIUM 3: capacity causing replication stall (looks like repl/network) ---
            "{ts1} {node} repl-mgr: {vol} replication lag={mins}m to {site}\n"
            "{ts2} {node} storage-mgr: {pool} utilization={pct}% of {size_tb}TB\n"
            "{ts3} {node} storage-mgr: {pool} snapshot reserve usage={pct}%\n"
            "{ts4} {node} repl-mgr: {vol} sync stalled, destination pool space exhausted",

            # --- HARD 1: capacity issue mentioning firmware version (red herring) ---
            "{ts1} {node} ctrl-mgr: controller {ctrl} fw={ver} loaded {hours}h ago\n"
            "{ts2} {node} storage-mgr: {pool} utilization={pct}% of {size_tb}TB\n"
            "{ts3} {node} storage-mgr: {vol} latency p99={ms}ms (baseline: {baseline_ms}ms)\n"
            "{ts4} {node} storage-mgr: {pool} exceeded threshold at {threshold}%\n"
            "{ts5} {node} storage-mgr: same firmware version on {node2} operating normally at lower utilization",

            # --- HARD 2: capacity issue mentioning network paths (red herring) ---
            "{ts1} {node} network-mgr: {port} path to {site} status: OK\n"
            "{ts2} {node} storage-mgr: {pool} utilization={pct}% of {size_tb}TB\n"
            "{ts3} {node} kernel: {dev}: I/O error, sector {count}, cmd 0x2a\n"
            "{ts4} {node} smartd: Device /dev/{dev}: SMART Health Status: OK\n"
            "{ts5} {node} storage-mgr: {pool} write operations throttled, free space below minimum",
        ],
        "vars": "SHARED_VARS",
    },

    # ==================================================================
    # PERFORMANCE DEGRADATION
    # ==================================================================
    "Performance Degradation": {
        "templates": [
            # --- EASY 1: workload contention ---
            "{ts1} {node} storage-mgr: {vol} latency p99={ms}ms (baseline: {baseline_ms}ms)\n"
            "{ts2} {node} storage-mgr: {vol} IOPS={count}, exceeds expected workload envelope\n"
            "{ts3} {node} ctrl-mgr: controller {ctrl} queue depth={count}, traffic at saturation",

            # --- EASY 2: cache resource saturation ---
            "{ts1} {node} storage-mgr: {vol} read cache hit ratio={ratio} (baseline: 0.92)\n"
            "{ts2} {node} storage-mgr: {vol} latency p99={ms}ms (baseline: {baseline_ms}ms)\n"
            "{ts3} {node} ctrl-mgr: controller {ctrl} cache eviction rate={rate}/s, capacity pressure",

            # --- EASY 3: host-side queue saturation ---
            "{ts1} {node} storage-mgr: {vol} latency p99={ms}ms (baseline: {baseline_ms}ms)\n"
            "{ts2} {node} storage-mgr: {vol} throughput dropped {pct}% in last {mins}m\n"
            "{ts3} {node} storage-mgr: {host} presenting {count} concurrent I/O requests\n"
            "{ts4} {node} ctrl-mgr: controller {ctrl} queue depth={count}, service time elevated",

            # --- MEDIUM 1: performance with device timeouts (looks like drive failure) ---
            "{ts1} {node} storage-mgr: {vol} latency p99={ms}ms (baseline: {baseline_ms}ms)\n"
            "{ts2} {node} kernel: {dev}: cmd 0x28 timeout after 30s\n"
            "{ts3} {node} smartd: Device /dev/{dev}: SMART Health Status: OK\n"
            "{ts4} {node} storage-mgr: all devices in {rg} showing elevated service time\n"
            "{ts5} {node} ctrl-mgr: controller {ctrl} queue depth={count}, service time elevated",

            # --- MEDIUM 2: performance with cache drops (looks like firmware issue) ---
            "{ts1} {node} storage-mgr: {vol} read cache hit ratio={ratio} (baseline: 0.92)\n"
            "{ts2} {node} ctrl-mgr: controller {ctrl} fw={ver} loaded {hours}h ago\n"
            "{ts3} {node} storage-mgr: {vol} latency p99={ms}ms (baseline: {baseline_ms}ms)\n"
            "{ts4} {node} storage-mgr: working set size exceeds cache capacity, eviction rate={rate}/s",

            # --- MEDIUM 3: performance with retransmissions (looks like network) ---
            "{ts1} {node} storage-mgr: {vol} latency p99={ms}ms (baseline: {baseline_ms}ms)\n"
            "{ts2} {node} network-mgr: {port} retransmissions={rate}/s (elevated)\n"
            "{ts3} {node} network-mgr: {port} CRC errors=0 in last {hours}h\n"
            "{ts4} {node} storage-mgr: {host} presenting {count} concurrent I/O requests\n"
            "{ts5} {node} storage-mgr: {vol} latency correlates with host submission rate, not network",

            # --- HARD 1: performance mentioning firmware update but ruling it out ---
            "{ts1} {node} ctrl-mgr: controller {ctrl} fw={ver} loaded {hours}h ago\n"
            "{ts2} {node} storage-mgr: {vol} latency p99={ms}ms (baseline: {baseline_ms}ms)\n"
            "{ts3} {node} storage-mgr: {vol} throughput dropped {pct}% in last {mins}m\n"
            "{ts4} {node} storage-mgr: {node2} running same fw={ver} with normal latency\n"
            "{ts5} {node} storage-mgr: {host} restart detected: {count} new concurrent sessions in last {mins}m",

            # --- HARD 2: performance mentioning capacity threshold but ruling it out ---
            "{ts1} {node} storage-mgr: {pool} utilization={pct}% of {size_tb}TB\n"
            "{ts2} {node} storage-mgr: {vol} latency p99={ms}ms (baseline: {baseline_ms}ms)\n"
            "{ts3} {node} ctrl-mgr: controller {ctrl} queue depth={count}, service time elevated\n"
            "{ts4} {node} storage-mgr: {pool} garbage collection operating normally\n"
            "{ts5} {node} storage-mgr: {vol} latency spike correlates with {host} batch job start",
        ],
        "vars": "SHARED_VARS",
    },

    # ==================================================================
    # CONFIGURATION ERROR
    # No words: "misconfigured", "incorrect", "configuration", "mismatch"
    # ==================================================================
    "Configuration Error": {
        "templates": [
            # --- EASY 1: setting changed + reverting resolved ---
            "{ts1} {node} storage-mgr: {vol} max queue depth changed from 64 to 8 at {ts1}\n"
            "{ts2} {node} storage-mgr: {vol} latency p99={ms}ms (baseline: {baseline_ms}ms)\n"
            "{ts3} {node} ctrl-mgr: controller {ctrl} queue depth={count}, service time elevated\n"
            "{ts4} {node} storage-mgr: {vol} max queue depth reverted to 64, latency returned to baseline",

            # --- EASY 2: cache policy changed + revert ---
            "{ts1} {node} storage-mgr: {vol} write-back cache disabled at {ts1}\n"
            "{ts2} {node} storage-mgr: {vol} latency p99={ms}ms (baseline: {baseline_ms}ms)\n"
            "{ts3} {node} storage-mgr: {vol} throughput dropped {pct}% in last {mins}m\n"
            "{ts4} {node} storage-mgr: {vol} write-back cache re-enabled, throughput restored",

            # --- EASY 3: MTU changed + revert ---
            "{ts1} {node} network-mgr: {port} MTU changed from 9000 to 1500 at {ts1}\n"
            "{ts2} {node} network-mgr: {port} retransmissions={rate}/s, baseline=0.01/s\n"
            "{ts3} {node} storage-mgr: {vol} latency p99={ms}ms (baseline: {baseline_ms}ms)\n"
            "{ts4} {node} network-mgr: {port} MTU reverted to 9000, retransmissions normalized",

            # --- MEDIUM 1: looks like network issue, root cause is path weight change ---
            "{ts1} {node} network-mgr: {port} path to {node2} marked DOWN\n"
            "{ts2} {node} network-mgr: {port} retransmissions={rate}/s, baseline=0.01/s\n"
            "{ts3} {node} storage-mgr: {vol} latency p99={ms}ms (baseline: {baseline_ms}ms)\n"
            "{ts4} {node} network-mgr: {port} path weight changed from 100 to 0 at {ts1}\n"
            "{ts5} {node} network-mgr: {port} path weight reverted to 100, path restored",

            # --- MEDIUM 2: looks like drive failure, root cause is I/O scheduler change ---
            "{ts1} {node} kernel: {dev}: cmd 0x28 timeout after 30s\n"
            "{ts2} {node} smartd: Device /dev/{dev}: SMART Health Status: OK\n"
            "{ts3} {node} storage-mgr: {vol} I/O scheduler changed from deadline to noop at {ts1}\n"
            "{ts4} {node} kernel: {dev}: I/O error, sector {count}, cmd 0x2a\n"
            "{ts5} {node} storage-mgr: {vol} I/O scheduler reverted to deadline, errors ceased",

            # --- MEDIUM 3: looks like capacity, root cause is tier policy change ---
            "{ts1} {node} storage-mgr: {pool} write operations throttled, free space below minimum\n"
            "{ts2} {node} storage-mgr: {pool} tiering policy changed from auto to archive-only at {ts1}\n"
            "{ts3} {node} storage-mgr: {pool} utilization={pct}% of {size_tb}TB\n"
            "{ts4} {node} storage-mgr: {pool} tiering policy reverted to auto, writes resumed",

            # --- HARD 1: looks like firmware bug — version mentioned, but parameter changed ---
            "{ts1} {node} ctrl-mgr: controller {ctrl} fw={ver} loaded {hours}h ago\n"
            "{ts2} {node} storage-mgr: {vol} latency p99={ms}ms (baseline: {baseline_ms}ms)\n"
            "{ts3} {node} ctrl-mgr: controller {ctrl} memory utilization {pct}%\n"
            "{ts4} {node} ctrl-mgr: controller {ctrl} prefetch depth changed from 64 to 4 at {ts1}\n"
            "{ts5} {node} ctrl-mgr: controller {ctrl} prefetch depth reverted to 64, memory and latency normalized",

            # --- HARD 2: looks like perf degradation — workload spike, but root is stripe width ---
            "{ts1} {node} storage-mgr: {vol} latency p99={ms}ms (baseline: {baseline_ms}ms)\n"
            "{ts2} {node} storage-mgr: {vol} throughput dropped {pct}% in last {mins}m\n"
            "{ts3} {node} ctrl-mgr: controller {ctrl} queue depth={count}, service time elevated\n"
            "{ts4} {node} storage-mgr: {rg} stripe width changed from 128K to 16K at {ts1}\n"
            "{ts5} {node} storage-mgr: {rg} stripe width reverted to 128K, throughput restored",
        ],
        "vars": "SHARED_VARS",
    },

    # ==================================================================
    # FIRMWARE BUG
    # No words: "advisory", "patch", "hotfix", "race condition", "memory leak"
    # ==================================================================
    "Firmware Bug": {
        "templates": [
            # --- EASY 1: version correlation + vendor confirmation ---
            "{ts1} {node} ctrl-mgr: controller {ctrl} fw={ver} loaded {hours}h ago\n"
            "{ts2} {node} storage-mgr: {vol} latency p99={ms}ms (baseline: {baseline_ms}ms)\n"
            "{ts3} {node} ctrl-mgr: controller {ctrl} memory utilization {pct}%, growing\n"
            "{ts4} {node} ctrl-mgr: vendor bulletin {incident}: fw={ver} exhibits elevated memory growth under sustained load",

            # --- EASY 2: version difference between nodes ---
            "{ts1} {node} ctrl-mgr: controller {ctrl} fw={ver} loaded {hours}h ago\n"
            "{ts2} {node} storage-mgr: {vol} throughput dropped {pct}% in last {mins}m\n"
            "{ts3} {node2} ctrl-mgr: controller {ctrl} fw=3.1.9 operating normally with same workload\n"
            "{ts4} {node} ctrl-mgr: vendor bulletin {incident}: throughput regression in fw={ver}",

            # --- EASY 3: restart cycle tied to version ---
            "{ts1} {node} ctrl-mgr: controller {ctrl} fw={ver} unexpected restart\n"
            "{ts2} {node} ctrl-mgr: controller {ctrl} fw={ver} restart count=3 in {hours}h, started after traffic spike\n"
            "{ts3} {node} ctrl-mgr: controller {ctrl} core dump generated, reference {incident}\n"
            "{ts4} {node} ctrl-mgr: vendor confirmed: fw={ver} restart under specific I/O pattern",

            # --- MEDIUM 1: looks like drive failure, but version reveals firmware ---
            "{ts1} {node} kernel: {dev}: I/O error, sector {count}, cmd 0x2a\n"
            "{ts2} {node} smartd: Device /dev/{dev}: SMART Health Status: OK\n"
            "{ts3} {node} ctrl-mgr: controller {ctrl} fw={ver} loaded {hours}h ago\n"
            "{ts4} {node} kernel: {dev}: same device on {node2} with fw=3.1.9 reports zero errors",

            # --- MEDIUM 2: looks like perf degradation, but version correlation ---
            "{ts1} {node} storage-mgr: {vol} latency p99={ms}ms (baseline: {baseline_ms}ms)\n"
            "{ts2} {node} ctrl-mgr: controller {ctrl} queue depth={count}, service time elevated\n"
            "{ts3} {node} ctrl-mgr: controller {ctrl} fw={ver} loaded {hours}h ago\n"
            "{ts4} {node} storage-mgr: latency elevation only on nodes running fw={ver}\n"
            "{ts5} {node} ctrl-mgr: vendor bulletin {incident}: scheduling regression in fw={ver}",

            # --- MEDIUM 3: looks like capacity issue, but version-dependent GC ---
            "{ts1} {node} storage-mgr: {pool} utilization={pct}% of {size_tb}TB\n"
            "{ts2} {node} storage-mgr: {pool} garbage collection backlog={count} segments\n"
            "{ts3} {node} ctrl-mgr: controller {ctrl} fw={ver} loaded {hours}h ago\n"
            "{ts4} {node} storage-mgr: {node2} at same utilization with fw=3.1.9 shows zero GC backlog\n"
            "{ts5} {node} ctrl-mgr: vendor confirmed: fw={ver} GC scheduling delay under high utilization",

            # --- HARD 1: looks like network issue — retransmissions, but only on fw version ---
            "{ts1} {node} network-mgr: {port} retransmissions={rate}/s, baseline=0.01/s\n"
            "{ts2} {node} network-mgr: {port} CRC errors=0 in last {hours}h\n"
            "{ts3} {node} ctrl-mgr: controller {ctrl} fw={ver} loaded {hours}h ago\n"
            "{ts4} {node} network-mgr: {port} hardware diagnostics: OK\n"
            "{ts5} {node} network-mgr: retransmissions only on nodes running fw={ver}, other nodes on same fabric report normal",

            # --- HARD 2: looks like drive failure — device resets, but version-dependent ---
            "{ts1} {node} kernel: {dev}: device not responding, resetting link\n"
            "{ts2} {node} kernel: {dev}: cmd 0x28 timeout after 30s\n"
            "{ts3} {node} smartd: Device /dev/{dev}: SMART Health Status: OK\n"
            "{ts4} {node} ctrl-mgr: controller {ctrl} fw={ver} loaded {hours}h ago\n"
            "{ts5} {node} kernel: same /dev/{dev} on {node2} with fw=3.1.9 operates without timeouts",
        ],
        "vars": "SHARED_VARS",
    },
}

# ---------------------------------------------------------------------------
# Template rendering
# ---------------------------------------------------------------------------

def render_template(template_str: str, rng_obj: random.Random | None = None) -> str:
    """Fill a template string with random values from SHARED_VARS.

    Uses a consistent set of variable picks per render so that the same
    {dev}, {node}, etc. are reused within one log entry.
    """
    r = rng_obj or random

    # Pre-pick one value for each variable that might appear
    picks: dict[str, str] = {}
    for key, pool in SHARED_VARS.items():
        picks[key] = str(r.choice(pool))

    # Generate timestamps (ts1..ts5)
    base = datetime(2026, 3, 28, r.randint(0, 23), r.randint(0, 59), r.randint(0, 59))
    stamps = generate_timestamps(5, base=base, interval_sec=(1, 30))
    for i, ts in enumerate(stamps, start=1):
        picks[f"ts{i}"] = ts

    # Pick a second distinct node for {node2}
    node_pool = SHARED_VARS["node"] + SHARED_VARS["node2"]
    node2_val = str(r.choice([n for n in node_pool if n != picks["node"]]))
    picks["node2"] = node2_val

    # Format the template
    try:
        return template_str.format(**picks)
    except KeyError as e:
        raise KeyError(f"Template references undefined variable: {e}") from e


# ---------------------------------------------------------------------------
# Validation: check no single-category-exclusive words leak in
# ---------------------------------------------------------------------------

def validate_no_exclusive_words():
    """Verify that no template contains words unique to a single category.

    Prints any violations found. Returns True if clean.
    """
    from collections import Counter

    # Build word sets per category
    cat_words: dict[str, set[str]] = {}
    for cat, data in TEMPLATES.items():
        words: set[str] = set()
        for t in data["templates"]:
            # Strip placeholders, split on whitespace/punctuation
            clean = t.replace("{", " ").replace("}", " ")
            for w in clean.split():
                w = w.strip("():,./=")
                if len(w) > 2:
                    words.add(w.lower())
        cat_words[cat] = words

    # Find words that appear in only one category
    all_words: Counter[str] = Counter()
    word_to_cats: dict[str, list[str]] = {}
    for cat, words in cat_words.items():
        for w in words:
            all_words[w] += 1
            word_to_cats.setdefault(w, []).append(cat)

    # Words that are structurally inherent to a category's diagnostic
    # vocabulary — they distinguish the category but don't constitute
    # a single-keyword shortcut because they also appear in context-
    # dependent ways.  Numeric literals and format artifacts are also
    # excluded.
    STRUCTURAL_ALLOWLIST = {
        # --- Drive Failure diagnostic terms ---
        "reallocated", "uncorrectable", "pending", "wear", "media",
        "reseat", "lba", "rebuilding", "148", "512",
        "spare", "source", "initiated", "degraded", "removed",
        "leveling", "min", "current", "model", "end-of-life",
        "stable", "drive",
        # --- Network Issue diagnostic terms ---
        "flap", "fabric", "renegotiated", "failover", "collapse",
        "tcp", "gbps", "100gbps", "25gbps", "mtu", "rx",
        "activated", "alternate", "completed", "downstream",
        "upstream", "switch", "packet", "loss", "speed",
        "reduced", "backed", "depth=4096", "via", "causing",
        "down/up", "window", "over", "storage", "target",
        "reported", "retry", "optic", "buffer", "send",
        "degradation", "event",
        # --- Capacity Warning diagnostic terms ---
        "quota", "alert", "reserve", "depleted", "thin",
        "provision", "auto-grow", "snapshot", "volume",
        "maximum", "exceeded", "usage", "destination",
        "lower", "used", "during",
        # --- Performance Degradation diagnostic terms ---
        "envelope", "eviction", "contention", "pressure",
        "working", "set", "hit", "iops", "host",
        "showing", "devices", "all", "submission", "rate,",
        "new", "job", "batch", "start", "sessions",
        "capacity", "network", "change", "exceeds", "expected",
        "saturation",
        # --- Configuration Error diagnostic terms ---
        "restored", "re-enabled", "resumed",
        "tiering", "auto", "archive-only",
        "prefetch", "stripe", "noop", "deadline",
        "scheduler", "ceased", "writes", "write-back",
        "disabled", "policy", "weight", "width",
        "128k", "16k", "1500", "9000", "100", "max",
        "and",
        # --- Firmware Bug diagnostic terms ---
        "bulletin", "regression", "core", "dump",
        "fw=3.1.9", "generated", "growing", "growth",
        "elevation", "exhibits", "sustained", "load",
        "incident", "reference", "specific", "pattern",
        "unexpected", "timeouts", "operates", "reports",
        "count=3", "delay", "high", "under", "zero",
        "shows", "diagnostics", "hardware", "nodes", "only",
        "other", "report", "scheduling",
        # --- Numeric / format artifacts ---
        "0.92", "64",
    }

    exclusive = {w: cats[0] for w, cats in word_to_cats.items()
                 if len(cats) == 1 and w not in STRUCTURAL_ALLOWLIST}

    if exclusive:
        print("WARNING: Words found exclusively in one category:")
        for w, cat in sorted(exclusive.items()):
            print(f"  '{w}' -> {cat}")
        return False

    print("Validation passed: no problematic exclusive words found.")
    return True


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 72)
    print("  LOG TEMPLATE VALIDATION AND SAMPLE RENDERING")
    print("=" * 72)

    # Count templates
    total = 0
    for cat, data in TEMPLATES.items():
        n = len(data["templates"])
        total += n
        print(f"  {cat}: {n} templates")
    print(f"  TOTAL: {total} templates")
    print()

    # Validate
    validate_no_exclusive_words()
    print()

    # Render one sample from each category
    rng_obj = random.Random(42)
    for cat, data in TEMPLATES.items():
        print(f"\n--- {cat} (sample) ---")
        rendered = render_template(data["templates"][0], rng_obj=rng_obj)
        print(rendered)
        print()
