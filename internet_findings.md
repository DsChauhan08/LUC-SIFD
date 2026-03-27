# Internet findings used

I could not directly scrape full Kaggle notebook cell contents from the code leaderboard pages without authenticated/session-backed access. Instead, I pulled publicly available related repositories and extracted practical ideas.

## Sources pulled
- `external/phucthai`: `phucthaiv02/Scientific-Image-Forgery-Detection`
- `external/mujahid`: `mujahidjadoon/Recod.ai-LUC---Scientific-Image-Forgery-Detection`
- `external/sumit`: `sumit1kr/scientific-image-forgery-detection`

## What to steal
- **Robust RLE handling** (from `external/phucthai/src/utils/rle.py`): strict encoding/decoding and validation habits.
- **Dual-task idea** (from `external/sumit/README.md`): jointly model authentic-vs-forged and pixel mask.
- **Noise/manipulation cue branch concept** (from `external/sumit/README.md`): copy-move benefits from handcrafted forensic cues plus learned cues.

## What to improve
- **CPU-first local verification**: add mock generation, mock evaluation, and deterministic scripts before Kaggle submission.
- **Explicit calibration loops**: tune authenticity thresholds and component filters with local CV, not fixed constants.
- **Runtime-aware model tiers**: fast heuristic baseline + trainable classifier + heavier Kaggle model branch.
