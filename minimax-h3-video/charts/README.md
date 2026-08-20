# MiniMax H3 charts

Current figures:

- [`1x-bf16-stage-breakdown.svg`](1x-bf16-stage-breakdown.svg), generated from the accepted CSVs by `../recipes/render-charts.py`;
- [`quality-seed1101-contact-sheet.jpg`](quality-seed1101-contact-sheet.jpg), five evenly spaced frames used for manual motion and repetition review;
- [`quality-seed1101-audio-spectrogram.png`](quality-seed1101-audio-spectrogram.png), the stereo soundtrack over the full 5.175-second file.

Additional figures will be generated after the remaining matrix is complete:

- end-to-end latency by topology and precision;
- generated video-seconds per wall-second;
- 1× versus 2× latency and aggregate-throughput scaling;
- peak HBM by configuration;
- BF16 versus FP8 quality deltas.
