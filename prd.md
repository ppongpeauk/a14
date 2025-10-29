# PRD: Audience-to-Studio Audio Diffusion

## 1) One-liner

A small PyTorch + Diffusers codebase that takes one or more audience recordings and produces denoised, dereverbated, de-roomed, acoustically neutralized “studio-master-like” audio.

## 2) Goals

* High impact, low code. Few files. Clear APIs.
* Strong enhancement on real concert bootlegs.
* Works with a single audience mic, and improves further with multiple audience mics of the same performance.
* Trains on synthetic pairs from clean music plus room simulation and, when available, real paired data.
* Runs on a single consumer GPU for both training and inference.
* Reproducible: seedable training, versioned configs, deterministic evaluation.

### Non-goals

* Text-to-audio generation.
* Full source separation.
* Perfect mastering chain. The output should be neutral and clean, not loudness-war ready.

## 3) Users and primary use cases

* Indie engineers restoring live sets.
* Creators improving phone captures for release.
* Researchers benchmarking diffusion enhancement.

## 4) System overview

The model operates in the time-frequency domain using mel spectrogram “images” with a UNet-style denoising backbone from Diffusers. Audio is mapped to mel spectrograms, processed, then vocoded back to waveform. This follows the Audio Diffusion approach that treats audio as images. ([Hugging Face][1])

### Key components

1. **Backbone**: `UNet2DConditionModel` from Diffusers with configurable channels.
2. **Noise scheduler**: DDPM or DDPMScheduler variants from Diffusers.
3. **Acoustic conditioning**: a small transformer encoder (e.g., AST or CLAP-like embedding) built with Hugging Face Transformers to summarize room/noise profile from the input mix and guide denoising.
4. **Mel front/back ends**: torchaudio for STFT/mel; HiFi-GAN or BigVGAN vocoder for waveform synthesis.
5. **Multi-recording fusion (optional)**: stack recordings as channels after alignment or use cross-attention over per-recording features.
6. **Training loop**: plain PyTorch + `accelerate`. Diffusers pipelines are used for inference composition; training is custom since pipelines are inference oriented. ([Hugging Face][2])

## 5) Data strategy

### Sources

* **Synthetic pairs**: take clean studio stems or mixes, convolve with real or simulated RIRs, add audience noise beds, PA leakage, band-limit, clipping. Target is the clean mix.
* **Real pairs**: audience capture aligned with FOH or multitrack. If alignment is rough, use beat-synchronous features for coarse sync.

### Augmentations

* Random RIR selection, late-reverb scaling, crowd noise SNR sweep, spectral tilt, mic distance roll-off, random band-stop, mobile mic saturation.

### Splits

* Artist-disjoint validation when possible.
* Hold out venues to test generalization to unseen rooms.

## 6) Model design

### Input and targets

* **Input**: mono or stereo audience waveform at 44.1 or 48 kHz.
* **Mel image**: 1024-point FFT, 75% overlap, 128 or 256 mel bins, log magnitude.
* **Condition vectors**: acoustic embedding from the same input, pooled over time and injected into UNet via cross-attention or FiLM.

### Losses

* **Diffusion noise-prediction MSE** (standard DDPM objective).
* **Aux STFT loss**: multi-resolution STFT magnitude loss to stabilize timbre and reduce musical noise. Widely used in enhancement work. ([arXiv][3])

### Metrics

* **FAD** for music quality, reference free. ([ISCA Archive][4])
* **SI-SDR** for distortion, **SRMR** for dereverb, **PESQ** for speech-heavy material.
* Human AB tests on 15 to 30 second clips.

### Schedulers

Expose `DDPM`, `DDIM`, and one fast sampler. Note that Diffusers has speed-quality tradeoffs documented in the schedulers guide. ([Hugging Face][1])

## 7) Inference flow

1. Load audio(s), resample, loudness-normalize.
2. Optional multi-track time alignment using cross-correlation on onset strength.
3. Compute mel spectrogram(s).
4. Compute acoustic conditioning.
5. Run diffusion denoising with chosen scheduler and steps.
6. Invert to waveform with vocoder.
7. Post-EQ tilt and light limiter off by default.

## 8) Repository layout (small and readable)

```
aud2studio/
  README.md
  pyproject.toml
  aud2studio/
    __init__.py
    config.py            # Pydantic config dataclasses
    data.py              # dataset + augment + RIR loader
    mel.py               # STFT/mel helpers
    cond.py              # acoustic embedding (Transformers)
    model.py             # UNet2DConditionModel wiring
    trainer.py           # train/eval loops with accelerate
    infer.py             # CLI inference
    metrics.py           # FAD, SI-SDR, SRMR hooks
    utils.py
  scripts/
    prepare_rirs.py
    make_synth_pairs.py
  tests/
    test_smoke.py
```

## 9) Config example (YAML)

```yaml
experiment: bootleg_v1
seed: 1337
audio:
  sample_rate: 48000
  clip_seconds: 10
  mono: true
mel:
  n_fft: 1024
  hop_length: 256
  n_mels: 128
  fmin: 20
  fmax: 20000
model:
  base_channels: 128
  layers: 4
  attention_resolutions: [16, 8, 4]
  conditioning_dim: 256
scheduler:
  name: ddpm
  num_inference_steps: 50
train:
  batch_size: 8
  lr: 2.0e-4
  ema: true
  grad_clip: 1.0
  epochs: 100
  optimizer: adamw
  fp16: true
data:
  rir_dir: ./data/rirs
  noise_dir: ./data/noise
  clean_music_dir: ./data/clean
  pairs_manifest: ./data/pairs.jsonl
eval:
  metrics: [si_sdr, srmr, fad]
  val_every_steps: 1000
export:
  half_precision: true
```

## 10) Minimal training loop (sketch)

```python
# trainer.py
import torch, math
from accelerate import Accelerator
from diffusers import UNet2DConditionModel, DDPMScheduler
from aud2studio.data import PairDataset, collate_audio
from aud2studio.cond import AcousticEncoder
from aud2studio.mel import wav_to_mel, mel_to_wav_vocoder

def train(cfg):
    accel = Accelerator(mixed_precision="fp16" if cfg.train.fp16 else "no")
    unet = UNet2DConditionModel(
        sample_size=(cfg.mel.n_mels, None),
        in_channels=1, out_channels=1, layers_per_block=2,
        block_out_channels=[cfg.model.base_channels*(2**i) for i in range(cfg.model.layers)]
    )
    cond_enc = AcousticEncoder(dim=cfg.model.conditioning_dim)
    noise_sched = DDPMScheduler()
    ds = PairDataset(cfg.data)
    opt = torch.optim.AdamW(unet.parameters(), lr=cfg.train.lr)

    unet, cond_enc, opt, dl = accel.prepare(unet, cond_enc, opt,
                                            torch.utils.data.DataLoader(ds, batch_size=cfg.train.batch_size,
                                                                        shuffle=True, collate_fn=collate_audio))
    unet.train()
    for step, batch in enumerate(dl):
        wav_in, wav_tgt = batch["aud"], batch["ref"]          # [B, T]
        mel_in = wav_to_mel(wav_in, cfg.mel)                  # [B, 1, M, N]
        mel_tgt = wav_to_mel(wav_tgt, cfg.mel)
        cond = cond_enc(mel_in)                               # [B, C]
        noise = torch.randn_like(mel_tgt)
        t = torch.randint(0, noise_sched.config.num_train_timesteps, (mel_tgt.size(0),), device=mel_tgt.device)
        noisy = noise_sched.add_noise(mel_tgt, noise, t)
        pred_noise = unet(noisy, t, encoder_hidden_states=cond).sample
        loss = torch.mean((pred_noise - noise)**2)
        accel.backward(loss)
        opt.step(); opt.zero_grad()
```

## 11) Inference CLI (sketch)

```bash
python -m aud2studio.infer \
  --inputs path/to/audience1.wav path/to/audience2.wav \
  --steps 50 --scheduler ddpm --save out.wav
```

```python
# infer.py
mel = wav_to_mel(wav_mix, cfg.mel)
cond = cond_enc(mel)
mel_clean = sample_with_scheduler(unet, cond, steps=cfg.scheduler.num_inference_steps)
wav_out = mel_to_wav_vocoder(mel_clean)
```

## 12) Training data preparation

* `scripts/prepare_rirs.py`: download RIR corpora, normalize, index.
* `scripts/make_synth_pairs.py`: convolve clean music with random RIRs, add noise, export JSONL manifest with paths and parameters.

## 13) Quality targets

* FAD ≤ 1.5 against a curated master reference set on validation clips. ([ISCA Archive][4])
* +6 dB SI-SDR over input.
* SRMR improvement ≥ 2.0 on dereverb benchmarks.
* 70 percent or more preference in blind AB tests versus strong non-diffusion baselines.

## 14) Performance targets

* Inference: 10 s clip in under 2.5 s on RTX 4060 Ti at 50 steps with fp16.
* Memory footprint: < 5 GB during inference with batch size 1, 128-bin mel, 50 steps.

## 15) Risks and mitigations

* **Hall coloration remains**: add a light post-EQ tilt and optional “dryness” control.
* **Over-smoothing**: include multi-resolution STFT loss, check transient preservation. ([arXiv][3])
* **Generalization to new venues**: widen RIR pool, randomize late-reverb decay, train with spectral tilts.
* **Pipeline-training mismatch**: keep training in raw PyTorch; wrap inference as a Diffusers `DiffusionPipeline` object for easy adoption. Pipelines are documented for inference. ([Hugging Face][2])

## 16) Milestones

* **M0**: Repo scaffold, mel I/O, dataset builder, smoke test with random weights.
* **M1**: Single-mic model trains on synthetic pairs; offline inference script; reaches initial SI-SDR and SRMR gains.
* **M2**: Add acoustic conditioning encoder and FAD metric.
* **M3**: Multi-mic fusion and fast sampler; export ONNX.
* **M4**: Human listening study and release v1 with model card.

## 17) Evaluation protocol

* Objective: SI-SDR, SRMR, PESQ on speech segments, and FAD on music segments. ([ISCA Archive][4])
* Subjective: 15 listeners, 20 tracks, MUSHRA-style UI.
* Report: per-genre breakdown, artifact taxonomy.

## 18) License and model card

* Code: Apache-2.0.
* Model weights: research-only if training on non-commercial data.
* Model card documents data sources, risks, and intended use.

## 19) Future extensions

* **Guided diffusion** with a lightweight noise model to sharpen details. ([arXiv][5])
* **Joint separation-enhancement**: add a small vocal/instrument mask to guide restoration.
* **Self-supervised pretraining** of the acoustic encoder on large music datasets.

---

### References for design choices

* Diffusers Audio Diffusion uses mel “image” representation. ([Hugging Face][1])
* Diffusers pipelines are for inference; training uses guides and custom loops. ([Hugging Face][2])
* FAD is a reference-free quality metric for music enhancement. ([ISCA Archive][4])
* Multi-resolution STFT losses stabilize enhancement models. ([arXiv][3])

[1]: https://huggingface.co/docs/diffusers/main/api/pipelines/audio_diffusion?utm_source=chatgpt.com "Audio Diffusion"
[2]: https://huggingface.co/docs/diffusers/en/api/pipelines/overview?utm_source=chatgpt.com "Pipelines"
[3]: https://arxiv.org/pdf/2303.14593?utm_source=chatgpt.com "arXiv:2303.14593v1 [cs.SD] 26 Mar 2023"
[4]: https://www.isca-archive.org/interspeech_2019/kilgour19_interspeech.pdf?utm_source=chatgpt.com "Fréchet Audio Distance: A Reference-free Metric for ..."
[5]: https://arxiv.org/html/2510.04157v1?utm_source=chatgpt.com "Diffusion-based Speech Enhancement with Noise Model ..."
