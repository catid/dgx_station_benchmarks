import csv
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "charts" / "network-throughput.png"
DATA = ROOT / "data" / "throughput.csv"

with DATA.open(newline="") as handle:
    rows = list(csv.DictReader(handle))

by_key = {
    (row["benchmark"], row["memory_path"], row["direction"], row["notes"]): float(row["bandwidth_gbps"])
    for row in rows
}
series = [
    ("Host RDMA\none-way", ("ib_write_bw", "host", "one_way", "4 QPs")),
    ("GPUDirect\none-way", ("ib_write_bw", "gpudirect_data_direct", "one_way", "4 QPs CUDA DMA-BUF Data Direct")),
    ("NCCL tuned\nall-reduce", ("nccl_all_reduce", "gpudirect_data_direct", "bus_bandwidth", "8 channels 4 QPs striped Ring Simple zero errors")),
    ("Host RDMA\nfull duplex", ("ib_write_bw", "host", "full_duplex_aggregate", "approximately 224 Gb/s each direction")),
    ("GPUDirect\nfull duplex", ("ib_write_bw", "gpudirect_data_direct", "full_duplex_aggregate", "391.53 Gb/s each direction")),
]
labels = [label for label, _ in series]
values = [by_key[key] for _, key in series]
colors = ["#64748b", "#22c55e", "#16a34a", "#64748b", "#22c55e"]

fig, ax = plt.subplots(figsize=(10, 5.4))
bars = ax.bar(labels, values, color=colors, width=0.68)
ax.axhline(400, color="#94a3b8", linestyle="--", linewidth=1, label="400GbE one-way line rate")
ax.axhline(800, color="#cbd5e1", linestyle=":", linewidth=1, label="800Gb/s full-duplex line rate")
ax.set_ylabel("Bandwidth (Gb/s)")
ax.set_title("DGX Station GB300: host memory vs ConnectX-8 Data Direct")
ax.set_ylim(0, 850)
ax.grid(axis="y", alpha=0.2)
ax.legend(loc="upper left")
for bar, value in zip(bars, values):
    label = Decimal(str(value)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    ax.text(bar.get_x() + bar.get_width() / 2, value + 13, str(label), ha="center", va="bottom", fontsize=10)
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=180)
