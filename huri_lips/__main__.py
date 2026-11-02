import argparse
import json
import sys
import time
import wave
from pathlib import Path

import numpy as np


def _read_wav(path: str):
    with wave.open(path) as w:
        width, channels, rate = w.getsampwidth(), w.getnchannels(), w.getframerate()
        raw = w.readframes(w.getnframes())
    if width != 2:
        raise SystemExit(f"{path}: 16-bit PCM WAV expected")
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples, rate


def _fetch(args) -> None:
    from huri_lips import lam

    if args.source:
        path = lam.import_checkpoint(Path(args.source))
    else:
        path = lam.fetch_checkpoint()
    print(path)


def _export(args) -> None:
    from huri_lips import lam
    from huri_lips.config import ONNX_FILE, ONNX_INT8_FILE, models_dir

    fp32 = models_dir() / ONNX_FILE
    if not fp32.exists() or args.force:
        print(f"exporting {fp32} ...")
        lam.export_onnx(fp32)
    if args.precision in ("int8", "both"):
        from onnxruntime.quantization import QuantType, quantize_dynamic

        int8 = models_dir() / ONNX_INT8_FILE
        print(f"quantizing {int8} ...")
        partial = int8.with_name(int8.name + ".part")
        quantize_dynamic(
            str(fp32),
            str(partial),
            weight_type=QuantType.QInt8,
            op_types_to_quantize=["MatMul", "Gemm"],
        )
        partial.replace(int8)
        if args.precision == "int8":
            fp32.unlink()
        print(int8)
    else:
        print(fp32)


def _run(args) -> None:
    from huri_events_lips import ARKIT_NAMES
    from huri_lips.engine import load_engine
    from huri_lips.stream import LipStream, to_16k

    samples, rate = _read_wav(args.wav)
    engine = load_engine(args.backend, device=args.device)
    stream = LipStream(engine)
    audio = to_16k(samples, rate)
    started = time.perf_counter()
    out = []
    for i in range(0, audio.size, 16000):
        out += stream.push(audio[i : i + 16000])
    out += stream.flush()
    took = time.perf_counter() - started
    frames = np.concatenate([w for _, w in out]) if out else np.zeros((0, 52))
    face = {
        "fps": 30,
        "names": list(ARKIT_NAMES),
        "frames": np.round(frames, 4).tolist(),
    }
    if args.output:
        Path(args.output).write_text(json.dumps(face))
    else:
        json.dump(face, sys.stdout)
    seconds = audio.size / 16000
    print(
        f"{engine.name}: {frames.shape[0]} frames for {seconds:.2f} s of speech "
        f"in {took:.2f} s ({seconds / max(took, 1e-9):.1f}x realtime)",
        file=sys.stderr,
    )


def _patch_voice(args) -> None:
    import shutil

    import onnx

    from huri_lips.config import models_dir

    voice = Path(args.voice)
    out = models_dir() / (voice.stem + ".aligned.onnx")
    model = onnx.load(str(voice))
    ceil = {
        o for node in model.graph.node if node.op_type == "Ceil" for o in node.output
    }
    if len(ceil) != 1:
        raise SystemExit(f"{voice}: expected one Ceil tensor, found {sorted(ceil)}")
    name = ceil.pop()
    if not any(o.name == name for o in model.graph.output):
        info = onnx.helper.ValueInfoProto()
        info.name = name
        model.graph.output.append(info)
    out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(out))
    shutil.copyfile(f"{voice}.json", f"{out}.json")
    print(out)


def _bench(args) -> None:
    from huri_lips.config import WINDOW_SAMPLES
    from huri_lips.engine import load_engine

    engine = load_engine(args.backend, device=args.device)
    rng = np.random.default_rng(0)
    for batch in (1, 2, 4):
        windows = rng.normal(0, 0.1, (batch, WINDOW_SAMPLES)).astype(np.float32)
        engine(windows)
        runs = max(1, int(args.seconds / 0.5 / batch))
        started = time.perf_counter()
        for _ in range(runs):
            engine(windows)
        each = (time.perf_counter() - started) / runs
        print(
            f"{engine.name} batch {batch}: {each * 1000:.0f} ms per call, "
            f"{batch * 0.5 / each:.1f}x realtime (0.5 s of face per window)"
        )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="python -m huri_lips")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fetch", help="download LAM's weights (once)")
    p.add_argument("--from", dest="source", help="a local copy of the archive")
    p.set_defaults(func=_fetch)

    p = sub.add_parser("export-onnx", help="export the model for onnxruntime")
    p.add_argument("--precision", choices=["int8", "fp32", "both"], default="int8")
    p.add_argument("--force", action="store_true", help="export again")
    p.set_defaults(func=_export)

    p = sub.add_parser("patch-voice", help="a Piper voice that reports phonemes")
    p.add_argument("voice", help="the Piper voice (.onnx, with its .onnx.json)")
    p.set_defaults(func=_patch_voice)

    for name, func, text in (
        ("run", _run, "a WAV file's ARKit weights as JSON"),
        ("bench", _bench, "time the model"),
    ):
        p = sub.add_parser(name, help=text)
        if name == "run":
            p.add_argument("wav")
            p.add_argument("-o", "--output", help="JSON file (default: stdout)")
        else:
            p.add_argument("--seconds", type=float, default=10.0)
        p.add_argument("--backend", choices=["auto", "torch", "onnx"], default="auto")
        p.add_argument("--device", help="torch device (cuda, cpu)")
        p.set_defaults(func=func)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
