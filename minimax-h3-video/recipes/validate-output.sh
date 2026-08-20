#!/usr/bin/env bash
set -Eeuo pipefail

[[ $# -eq 2 ]] || {
  echo "Usage: $0 OUTPUT.mp4 VALIDATION_DIR" >&2
  exit 2
}

video="$1"
validation_dir="$2"
[[ -f "${video}" ]] || {
  echo "Video not found: ${video}" >&2
  exit 2
}
mkdir -p "${validation_dir}"

ffprobe -v error \
  -count_frames \
  -show_entries \
    format=duration,size,bit_rate:stream=index,codec_type,codec_name,width,height,r_frame_rate,nb_frames,nb_read_frames,sample_rate,channels,channel_layout,duration \
  -of json "${video}" >"${validation_dir}/ffprobe.json"

ffmpeg -v error -xerror -i "${video}" \
  -map 0:v:0 -map 0:a:0 -f null -

ffmpeg -hide_banner -i "${video}" \
  -vf 'blackdetect=d=0.5:pix_th=0.10,freezedetect=n=-60dB:d=1' \
  -an -f null - \
  2>"${validation_dir}/video-warnings.log" || true

ffmpeg -hide_banner -i "${video}" \
  -vn -af 'silencedetect=n=-50dB:d=0.25,astats=metadata=1:reset=1' \
  -f null - \
  2>"${validation_dir}/audio-warnings.log" || true

sha256sum "${video}" >"${validation_dir}/sha256.txt"
echo "Decode passed. Review ffprobe.json and both warning logs before publication."
