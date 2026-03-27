# Ranked steal vs improve matrix

This ranking is based on the public notebook names and common high-performing patterns for this competition type, then converted into a CPU-first implementation plan.

## 1) DINOv2 feature encoder idea (highest value)
- Steal: strong pretrained visual features are robust to biomedical texture variety.
- Improve: add explicit copy-move matching signal (self-similarity map), because generic encoders alone do not explicitly model duplicated regions.
- Why this matters: copy-move is relational; a pure segmentation encoder may miss subtle duplicated pairs.

## 2) Area-ratio postprocess idea (high value)
- Steal: simple area and component heuristics can remove many false positives.
- Improve: tune thresholds by validation folds and include min/max component area, component count cap, and morphology sequence.
- Why this matters: competition metric is F1-like; reducing false positives is often the fastest gain.

## 3) Strong thresholding calibration (high value)
- Steal: single score thresholding for authentic vs forged decision.
- Improve: two-stage thresholding: (a) pixel-level mask threshold, (b) image-level authenticity threshold from suspicious area fraction.
- Why this matters: many images are authentic; calibrated authenticity detection is critical.

## 4) Duplicate-region pairing constraints (medium-high value)
- Steal: candidate regions from repeated local patterns.
- Improve: enforce geometric plausibility (consistent offset / patch similarity confirmation) to cut repeated natural textures.
- Why this matters: biomedical images have repetitive motifs that create false matches.

## 5) Lightweight inference profile (medium value)
- Steal: compact pipeline suitable for competition runtime.
- Improve: explicit CPU profile mode (fixed max side, stride/patch size controls), so local benchmarking maps to Kaggle notebook limits.
- Why this matters: predictable runtime encourages more experiments and stable submissions.

## 6) Submission robustness (must-have)
- Steal: exact RLE formatting and `authentic` string handling.
- Improve: add round-trip RLE checks and strict CSV writer so no formatting errors at submission time.
- Why this matters: malformed output kills scores regardless of model quality.
