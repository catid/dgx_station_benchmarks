# P1: excluded cold-cache FlashInfer CUTLASS startup

This is failure provenance, not a performance result. The exact P0 workload
and runtime were retained while changing only the NVFP4 MoE backend to
`FLASHINFER_CUTLASS`. Both ranks loaded the model and completed the first
backend-specific compile, but startup failed the strict long-context capacity
gate before the API became healthy. No benchmark request was issued.

The named containers were stopped gracefully, the current-boot kernel scan was
clean, and HBM returned to the idle baseline. The populated compile cache was
retained for one separately recorded warm-cache validation; this directory
continues to describe only the excluded cold-cache attempt.

