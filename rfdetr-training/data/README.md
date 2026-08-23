# RF-DETR training data artifacts

`raw/control-metrics.csv` and `raw/optimized-metrics.csv` are the authoritative
Lightning CSV logs. The final row containing validation fields supplies the
reported regular and EMA mAP values. The timing JSON files are emitted by
`train_large.py` after `model.train()` returns.

`raw/sweep-results.tsv` contains every successful synchronized sweep cell.
`raw/optimized-training-config.json` is RF-DETR's complete resolved
configuration. The two NCCL logs show both HCAs selected at 400,000 Mb/s and
merged for the distributed run.

The optimized rank-0 checkpoints were retained on the benchmark system but are
not committed:

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `checkpoint_best_regular.pth` | 276,969,975 | `d620be0a165558206dbbab2312431774b1d03c2822f205dde5d07b10dc7b2ee9` |
| `checkpoint_best_ema.pth` | 138,548,891 | `003c110b0f6c2e1a777d7ebbcc921de563289c11b57d367fb65ef1e024104c34` |
| `checkpoint_best_total.pth` | 138,536,283 | `3a83af7712a97718f75db8f2fd9135b832c077c7a22e067a6a545c280a8b3144` |

Core environment:

- Ubuntu kernel `6.17.0-1029-nvidia-64k`, ARM64
- NVIDIA driver 595.84
- One NVIDIA GB300 with 256,703 MiB reported memory per host
- Python 3.12.3
- RF-DETR 1.9.3
- PyTorch `2.15.0.dev20260821+cu132`
- Torchvision `0.30.0.dev20260822+cu132`
- PyTorch Lightning 2.6.5
- Pillow 12.3.0 and pycocotools 2.0.11
- Kornia 0.8.3 only for its optional sweep cell

Both hosts returned to their normal idle-HBM baselines after the run, had no
compute clients, and had no Xid, ATS/PMA, RM initialization, kernel-oops, or
soft-lockup signatures in the boot kernel journal.
