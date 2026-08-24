# Safety stop: retained HBM before first FSDP launch

The journal-first preflight on 2026-08-23 found no current-boot kernel danger
signature on either host. The subsequent idle check found:

- `gemini2`: 54,944 MiB used on the GB300 versus its 793 MiB normal baseline.
- `gemini1`: 1,063 MiB used on the GB300 versus its 67 MiB normal baseline.
- No running containers on either host.
- No compute process in `nvidia-smi` or `pmon` on either host.
- No UVM client in `fuser` or `lsof`; only `nvidia-persistenced` had the
  per-GPU device open.

No training process was launched. No memory setting was raised, no retry was
made, and no GPU reset, PCI operation, driver reload, or reboot was attempted.
Per `/home/catid/frontier-bench/SAFETY.md`, GPU work must remain stopped until
the user coordinates a normal reboot of both hosts.

Raw evidence is under
`results/preflight-20260823T1715Z/{gemini1,gemini2}/` on the corresponding
hosts.
