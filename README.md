# aud2studio

Small, readable PyTorch + Diffusers project that turns audience recordings into cleaner, more "studio-like" audio.

## Install

```bash
pip install -e .
```

If you hit a SciPy/NumPy ABI error on HPC (e.g., "numpy.dtype size changed"), use constraints:

```bash
pip install -e . -c constraints/py39.txt
```

If you see ImportError: cannot import name 'cached_download' from 'huggingface_hub', pin hub to 0.19.4:

```bash
pip install --upgrade --force-reinstall huggingface_hub==0.19.4
```

### Latest stack (recommended)
- For the newest `diffusers`/`transformers`/`huggingface_hub`, use Python 3.10+ to avoid SciPy/NumPy wheel gaps on 3.9.

Fresh env with latest libs:
```bash
python3.10 -m venv .venv && source .venv/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install -e .
# Ensure the newest bits
python -m pip install -U diffusers transformers huggingface_hub accelerate
```

## Your data layout (one-folder pairs)
You said all pairs live in a single folder, with files like:
- `input0_0.wav` → ground truth (the suffix `_0` is the reference)
- `input0_1.wav`, `input0_blah.wav` → audience inputs for the same group
- Works with `.wav` or `.mp3`

Example:
```
/path/to/data/
  input0_0.wav
  input0_1.wav
  input0_blah.wav
  input1_0.mp3
  input1_1.mp3
  input1_2.mp3
```

Generate a JSONL manifest from this folder:
```bash
python scripts/make_manifest_from_folder.py \
  --data_dir /path/to/data \
  --out ./data/pairs.jsonl
```
Each JSONL line looks like:
```json
{"aud": ["/abs/path/input0_1.wav", "/abs/path/input0_blah.wav"], "ref": "/abs/path/input0_0.wav"}
```

## Train (toy)
```bash
aud2studio-train --config examples/config.yaml
```
- Edit `examples/config.yaml` to point `data.pairs_manifest` to your generated file.
- Defaults are light for a quick smoke run; increase batch size/epochs later.

## Inference
Single or multi-inputs are supported. The model denoises in mel space and inverts with a Griffin-Lim fallback vocoder.
```bash
aud2studio-infer \
  --inputs /path/to/data/input0_1.wav /path/to/data/input0_blah.wav \
  --save out.wav \
  --config examples/config.yaml
```

aud2studio-infer --inputs ./data/end_1.mp3 --save out.wav --config examples/config.yaml

## Repo map
```
aud2studio/
  config.py      # Pydantic config + YAML
  data.py        # JSONL pairs dataset (+ mp3/wav via torchaudio)
  mel.py         # mel I/O + Griffin-Lim fallback
  cond.py        # acoustic conditioning encoder (Transformer)
  model.py       # UNet2DConditionModel wiring (Diffusers)
  trainer.py     # accelerate-based training loop (DDPM)
  infer.py       # CLI inference
  metrics.py     # SI-SDR implemented; SRMR/FAD optional
scripts/
  make_manifest_from_folder.py  # build pairs.jsonl from your folder
  prepare_rirs.py               # index RIRs
  make_synth_pairs.py           # synth bootstrapping (optional)
examples/
  config.yaml
```

## Notes
- For .mp3 input, `torchaudio` must be built with FFmpeg support.
- Griffin-Lim is a quality fallback; swap BigVGAN/HiFi-GAN later for better timbre.
- SRMR/FAD are optional; SI-SDR is built-in for quick checks.
