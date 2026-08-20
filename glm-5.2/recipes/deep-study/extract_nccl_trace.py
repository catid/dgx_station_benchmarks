#!/usr/bin/env python3
"""Extract replayable collective sizes from short NCCL TRACE logs."""

from __future__ import annotations

import argparse
import collections
import re
from pathlib import Path


OP_RE = re.compile(r"\b(AllReduce|AllGather|ReduceScatter|Broadcast|Reduce|Send|Recv)\b", re.I)
COUNT_RE = re.compile(r"\bcount\s*[=: ]\s*([0-9]+)", re.I)
DTYPE_RE = re.compile(r"\b(?:datatype|dtype)\s*[=: ]\s*([A-Za-z0-9_]+)", re.I)
TYPE_BYTES = {
    "0": 1,
    "1": 1,
    "2": 4,
    "3": 4,
    "4": 8,
    "5": 8,
    "6": 2,
    "7": 4,
    "8": 8,
    "9": 2,
    "ncclint8": 1,
    "nccluint8": 1,
    "ncclint32": 4,
    "nccluint32": 4,
    "ncclint64": 8,
    "nccluint64": 8,
    "ncclfloat16": 2,
    "ncclhalf": 2,
    "ncclfloat32": 4,
    "ncclfloat": 4,
    "ncclfloat64": 8,
    "nccldouble": 8,
    "ncclbfloat16": 2,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    args = parser.parse_args()
    rows: collections.Counter[tuple[str, int, str, int]] = collections.Counter()
    for path in args.logs:
        for line in path.read_text(errors="replace").splitlines():
            op_match = OP_RE.search(line)
            count_match = COUNT_RE.search(line)
            dtype_match = DTYPE_RE.search(line)
            if not (op_match and count_match and dtype_match):
                continue
            operation = op_match.group(1)
            count = int(count_match.group(1))
            dtype = dtype_match.group(1)
            width = TYPE_BYTES.get(dtype.casefold())
            if width is None or count <= 0:
                continue
            rows[(operation, count, dtype, count * width)] += 1
    if not rows:
        raise SystemExit(
            "No replayable NCCL TRACE collectives found. Preserve the raw logs and use "
            "a short Nsight Systems NCCL trace to obtain payload sizes."
        )
    print("operation\tcount\tdatatype\tbytes\toccurrences")
    for (operation, count, dtype, size), occurrences in sorted(rows.items(), key=lambda row: (row[0][0], row[0][3])):
        print(f"{operation}\t{count}\t{dtype}\t{size}\t{occurrences}")


if __name__ == "__main__":
    main()
