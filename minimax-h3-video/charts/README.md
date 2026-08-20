# MiniMax H3 charts

Current figures:

- [`1x-bf16-stage-breakdown.svg`](1x-bf16-stage-breakdown.svg), generated from the accepted CSVs by `../recipes/render-charts.py`;
- [`duration-scaling.svg`](duration-scaling.svg), one-station client latency for the measured native durations, with the unsupported patched 30-second row colored orange;
- [`15s-independent-throughput.svg`](15s-independent-throughput.svg), measured complete clips/hour for one versus two independent resident replicas;
- [`quality-seed1101-contact-sheet.jpg`](quality-seed1101-contact-sheet.jpg), five evenly spaced frames used for manual motion and repetition review;
- [`quality-seed1101-audio-spectrogram.png`](quality-seed1101-audio-spectrogram.png), the stereo soundtrack over the full 5.175-second file.

All SVGs are deterministically rebuilt from the checked-in CSVs by
`../recipes/render-charts.py`. Animated sample previews and contact sheets live
under [`../samples/`](../samples/README.md).
