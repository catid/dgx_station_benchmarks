#!/usr/bin/env bash

set -Eeuo pipefail

if (( $# != 2 )); then
    echo "usage: $0 OPENIMAGES_ROOT RFDETR_DATASET_DIR" >&2
    exit 2
fi

source_root=$(realpath -e -- "$1")
dest_root=$(realpath -m -- "$2")

prepare_source_split() {
    local source_split=$1
    local data_dir="$source_root/$source_split/data"
    local annotation="$source_root/$source_split/labels/openimages-mlperf.json"
    local expected_link="../labels/openimages-mlperf.json"

    test -d "$data_dir"
    test -s "$annotation"
    if [[ -L "$data_dir/_annotations.coco.json" ]]; then
        test "$(readlink -- "$data_dir/_annotations.coco.json")" = "$expected_link"
    elif [[ -e "$data_dir/_annotations.coco.json" ]]; then
        echo "refusing to replace $data_dir/_annotations.coco.json" >&2
        exit 1
    else
        ln -s "$expected_link" "$data_dir/_annotations.coco.json"
    fi
}

link_destination_split() {
    local dest_split=$1
    local source_split=$2
    local target="$source_root/$source_split/data"
    local link="$dest_root/$dest_split"

    if [[ -L "$link" ]]; then
        test "$(readlink -f -- "$link")" = "$target"
    elif [[ -e "$link" ]]; then
        echo "refusing to replace $link" >&2
        exit 1
    else
        ln -s "$target" "$link"
    fi
}

prepare_source_split train
prepare_source_split validation
mkdir -p "$dest_root"
link_destination_split train train
link_destination_split valid validation
link_destination_split test validation

python3 - "$dest_root" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
expected = {
    "train": (1_170_301, 264),
    "valid": (24_781, 264),
}
for split, (expected_images, expected_categories) in expected.items():
    annotation = root / split / "_annotations.coco.json"
    with annotation.open("rb") as handle:
        payload = json.load(handle)
    actual = (len(payload["images"]), len(payload["categories"]))
    wanted = (expected_images, expected_categories)
    if actual != wanted:
        raise SystemExit(f"{split}: expected {wanted}, found {actual}")
    print(f"{split}: images={actual[0]} categories={actual[1]} PASS")
PY
