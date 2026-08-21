# P1: excluded warm-cache FlashInfer CUTLASS startup

This is failure provenance, not a performance result. It is the one controlled
cache-hit validation authorized after the cold-cache attempt populated its
backend-specific AOT artifacts and returned both GPUs to a clean idle state.
No workload, utilization, checkpoint, runtime, or topology flag changed.

Both ranks loaded the cached graph successfully, but the long-context capacity
gate remained asymmetric: node 0 exposed 2.19 GiB for KV while node 1 exposed
10.39 GiB. The configured 135,168-token ceiling needs 6.0 GiB on every rank,
so the API never became healthy and no benchmark request was issued. There was
no further retry. Graceful teardown, the current-boot kernel scan, and the idle
HBM gate all passed.

