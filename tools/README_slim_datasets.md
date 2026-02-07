# Slim Dataset Generator

The original repository included several gigabytes of training and optimization
artefacts. To keep the public release lean while still allowing end-to-end
execution of the pipelines, lightweight placeholder datasets can be generated
with the helper script:

```bash
python tools/generate_slim_datasets.py
```

This produces tiny synthetic files (<100 KB each) that mimic the structure of
the original assets for:

- `optimal` grid-world experiments (grid sizes 6 and 13)
- `suboptimal` agent studies (grid 6)
- `overcooked-ai` simulations (grid 6)
- Human validation notebooks

The artefacts are strictly for demonstration/testing purposes and are not the
original experimental results. They allow the training and plotting scripts to
run without errors after the heavy data has been trimmed.
