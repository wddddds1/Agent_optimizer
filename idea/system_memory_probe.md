# System Memory Probe and Bandwidth Microbench (Idea)

Goal: extend the platform probe with deeper memory details and an optional
lightweight bandwidth check. This is intentionally non-blocking and can be
added later.

Why:
- Thread scaling and cache/NUMA choices depend on memory bandwidth and latency.
- Knowing memory type (DDR4/DDR5/LPDDR), channels, and bandwidth can guide
  the parallel candidate generator (e.g., thread caps).

Scope (Phase 1):
- Keep the existing probe (total RAM, page size, cache, cores).
- Add deeper memory metadata and an optional microbench that runs only on
  baseline runs or on demand.

Proposed fields (new):

```
memory:
  total_bytes
  page_size
  type               # DDR4/DDR5/LPDDR (if discoverable)
  speed_mt_s         # memory data rate (if discoverable)
  channels           # channel count (if discoverable)
  modules            # count of DIMMs (if discoverable)
  bandwidth:
    read_gbps
    write_gbps
    copy_gbps
    size_bytes       # working set size for microbench
    duration_ms
```

OS-specific sources (best-effort):
- macOS:
  - `system_profiler SPMemoryDataType` (memory type, speed)
  - `sysctl hw.memsize`, `sysctl hw.pagesize`
  - perf levels already used for P/E core counts
- Linux (Ubuntu/CentOS/Rocky):
  - `/proc/meminfo` for total RAM
  - `/sys/devices/system/node` for NUMA nodes
  - `dmidecode -t memory` or `lshw -class memory` for type/speed
    (may require elevated privileges)

Microbench idea (optional):
- A minimal, self-contained C++ or Python memory bandwidth test:
  - Allocate a large buffer (e.g., 256–512MB)
  - Run streaming read/write/copy loops
  - Report GB/s
- Run once per session if enabled:
  - Default: off
  - Enable via config flag `probe.memory_benchmark: true`

Safety considerations:
- Keep the benchmark short (e.g., <200ms).
- Avoid huge allocations on low-memory systems.
- Record the result in the platform probe for the agent.

Integration points:
- `skills/system_probe.py`: add memory metadata & optional microbench
- `configs/planner.yaml` or new `configs/probe.yaml`: opt-in flag
- Expose in ParameterExplorerAgent prompt (platform probe section)
