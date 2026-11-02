# huri-module-lips-sync

Lip sync for HuRI: the face that goes with the voice. The `lips` module hears
what HuRI says and streams **ARKit blend shape weights** (jawOpen,
mouthFunnel, mouthClose, ... 52 of them, 30 frames a second) in step with the
voice, for an avatar or a robot face to play. The weights are computed by
[LAM Audio2Expression](https://github.com/aigc3d/LAM_Audio2Expression)
(Alibaba 3DAIGC, Apache-2.0), a learned model: nothing is pre-recorded, and
it runs **on a GPU or on a CPU**.

```
huri.audio.out ──> lips ──> lips.face   (FaceWeights: 52 ARKit weights at 30 fps,
                                         pts on the voice's timeline)
```

| Module | Topics | What it does |
|---|---|---|
| `lips` | `huri.audio.out` → `lips.face` | ARKit blend shapes from the voice (TTS, and huri-module-think's thinking phrases) |

`lips.face` is this repository's own event (`huri-events-lips`, in `events/`):
a client installs only that to decode it. Its `pts` is the clock of
`Audio.pts`, so a client plays frame `i` when the voice reaches
`pts + i / fps`. The frames come ahead of the voice: a sentence's face is
computed while it is still being played.

## GPU or CPU

The model (wav2vec 2.0 base and a small decoder, 102 M parameters) runs on
2.13 s windows of audio, each giving the next 0.5 s of face. Two backends
compute the same thing:

| Backend | Where | Measured (0.5 s of face per window) |
|---|---|---|
| `torch` | NVIDIA (CUDA), AMD (ROCm) or CPU | RX 7700S (ROCm 6.4): 22 ms, 23x realtime; CPU: 71 ms, 7x |
| `onnx` | CPU, onnxruntime, int8 weights (145 MB) | 16-core x86 laptop CPU: 51 ms, 10x realtime |

`HURI_LIPS_BACKEND=auto` (the default) takes torch on a GPU, else the ONNX
export when it is there, else torch on the CPU. The int8 export differs from
the float model by 0.0015 on average (jawOpen within 0.015); the GPU and CPU
float models agree within 0.0001. On NVIDIA it is not tested yet.

## Installing and testing it

In a virtualenv, install HuRI's core and curated events, then the module:

```sh
python -m venv .venv && . .venv/bin/activate
python -m pip install \
  "huri-core[server] @ git+https://github.com/Lucy-Robotics/huri-core.git@<ref>" \
  "huri-events @ git+https://github.com/Lucy-Robotics/huri-modules.git@<ref>#subdirectory=packages/huri-events"
./install.sh             # picks requirements-{cpu,nvidia,amd}.txt, installs events/ and the module,
                         # downloads LAM's weights and, on a CPU, exports them to int8 ONNX
pytest
```

`./install.sh --no-models` installs the packages only; the model then comes
with:

```sh
python -m huri_lips fetch                     # LAM's weights, 373 MB, once
python -m huri_lips fetch --from LAM_audio2exp_streaming.tar   # or from a local copy
python -m huri_lips export-onnx               # int8 ONNX for the CPU (--precision fp32|both)
```

Weights and exports go to `$HURI_LIPS_HOME` (default `~/.cache/huri-lips`).
To try the model on a WAV file, or time it:

```sh
python -m huri_lips run speech.wav -o face.json   # {"fps", "names": [52], "frames": [[52], ...]}
python -m huri_lips bench --backend onnx
```

## Using it in HuRI

On the server, enable it next to a TTS:

```sh
HURI_MODULES=...,tts,lips
```

and give it a deployment in the server config, like the other model modules:

```yaml
- name: LipsHandle
  ray_actor_options:
    num_cpus: 1
    num_gpus: 0.1   # 0 on a CPU-only machine
```

Its settings come from the environment:

| Variable | Default | What it sets |
|---|---|---|
| `HURI_LIPS_BACKEND` | `auto` | `auto`, `torch` or `onnx` |
| `HURI_LIPS_DEVICE` | `cuda` if any | torch device |
| `HURI_LIPS_HALF` | `0` | `1`: float16 on a GPU |
| `HURI_LIPS_THREADS` | all cores | onnxruntime threads |
| `HURI_LIPS_IDENTITY` | `0` | speaking style, 0..11 (the model's training speakers) |
| `HURI_LIPS_HOME` | `~/.cache/huri-lips` | weights and exports |

A client asks for it with the TTS:

```yaml
modules:
  tts: { name: tts }
  lips: { name: lips, args: { hop_frames: 15, lookahead_frames: 6 } }
subscribe: [huri.audio.out, lips.face]
```

| Argument | Default | What it does |
|---|---|---|
| `hop_frames` | 15 | frames per window (0.5 s): smaller, the first frames come sooner |
| `lookahead_frames` | 6 | audio after its frames each window hears (0.2 s): the mouth anticipates |
| `max_batch` | 4 | most windows per model call (a long sentence) |
| `silence_rms` | 0.001 | quieter frames are silence: the mouth closes after 7 of them |
| `smooth_frames` | 0 | Savitzky-Golay window (LAM's 5 smooths away quick lip closures) |
| `symmetric` | true | same weight for the left and right shapes (not the gaze) |
| `delay_s` | 0.13 | frames shown this much later than LAM puts them: its mouth leads the voice (below) |

LAM's mouth leads the voice. Measured against the phoneme timings of a Piper
voice (en_US-lessac-medium, 8 sentences), its frames match the sounds best
0.13 s later: the correlation of the lip gap with the sounds goes from -0.01
to 0.51, and the lips close on 80% of "p", "b" and "m" instead of 29%. LAM
run offline on whole sentences leads as much, so it is the model's, not the
streaming's; `delay_s` shows each frame that much later.

The weights are ARKit's, as most avatar rigs name their blend shapes; a
client that drives its own blinks or gaze uses the mouth, jaw and cheek
shapes and leaves the eyes to itself. HuRI-Playground does so.

## What is in it

| Path | What it is |
|---|---|
| `huri_lips/module.py` | `Lips`, the module (utterances, timeline), and `LipsHandle`, the deployment |
| `huri_lips/stream.py` | the windows and LAM's post-processing, numpy only |
| `huri_lips/engine.py` | the backends: torch (GPU/CPU), onnxruntime (CPU) |
| `huri_lips/lam.py` | LAM-A2E's network in PyTorch, its weights, the ONNX export |
| `huri_lips/__main__.py` | `fetch`, `export-onnx`, `run`, `bench` |
| `events/` | `huri-events-lips`: the `lips.face` event (`FaceWeights`) and the ARKit names |
| `tests/` | the event's wire format, the windows, the module with a fake model, the server's checks |

## Licence

GPL-3.0 (see LICENSE). LAM Audio2Expression's code and weights are
Apache-2.0: see NOTICE for what comes from it.
