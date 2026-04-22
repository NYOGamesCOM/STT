#!/usr/bin/env python3
"""
STT transcription benchmark.

Runs each (wav, reference transcript) pair through the given model(s) and
reports latency, real-time factor, and word error rate.

    python scripts/benchmark.py benchmarks/clips
    python scripts/benchmark.py benchmarks/clips --model base
    python scripts/benchmark.py benchmarks/clips --model all --out BENCHMARKS-run.md

Folder layout: one `.wav` per clip. A matching `.txt` file next to a wav
(same stem) is used as the reference transcript for WER. Without a .txt
file, latency/RTF are still reported; WER is skipped.
"""

from __future__ import annotations

import argparse
import os
import platform
import re
import statistics
import sys
import time
from pathlib import Path

# Make "stt" importable when run from anywhere
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from stt import APP_VERSION, Transcriber        # noqa: E402

MODELS = ["tiny", "base", "small"]


# ── WER ──────────────────────────────────────────────────────────────────────

def _normalise(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s']", " ", text)
    return text.split()


def word_error_rate(reference: str, hypothesis: str) -> float:
    """
    Levenshtein (word-level) / len(reference).
    0.0 = perfect; 1.0 = every word wrong. Can exceed 1.0 if the
    transcription is much longer than the reference (lots of inserts).
    """
    ref = _normalise(reference)
    hyp = _normalise(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0

    # Standard DP edit distance
    m, n = len(ref), len(hyp)
    prev = list(range(n + 1))
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev, curr = curr, prev
    return prev[n] / m


# ── Single-clip measurement ──────────────────────────────────────────────────

def measure_clip(tx: Transcriber, wav: Path, ref: str | None) -> dict:
    """Run the model on one wav file and return measured numbers."""
    t0 = time.perf_counter()
    segments, info = tx._model.transcribe(        # noqa: SLF001
        str(wav),
        language="en",
        beam_size=1,
        vad_filter=False,
        condition_on_previous_text=False,
        without_timestamps=True,
    )
    # Force iteration so decoding actually completes before timing stops
    text = " ".join(s.text.strip() for s in segments).strip()
    elapsed = time.perf_counter() - t0

    duration = float(getattr(info, "duration", 0.0)) or _wav_duration(wav)
    rtf = elapsed / max(duration, 1e-6)
    wer_val = word_error_rate(ref, text) if ref else None

    return {
        "file":      wav.name,
        "audio_s":   duration,
        "latency_s": elapsed,
        "rtf":       rtf,
        "wer":       wer_val,
        "text":      text,
        "ref":       ref,
    }


def _wav_duration(path: Path) -> float:
    """Fallback WAV duration without external deps (stdlib wave)."""
    import wave
    try:
        with wave.open(str(path), "rb") as f:
            return f.getnframes() / f.getframerate()
    except Exception:
        return 0.0


# ── Reporting ────────────────────────────────────────────────────────────────

def fmt_pct(x: float | None) -> str:
    return "—" if x is None else f"{x * 100:.1f}%"


def print_summary(rows_by_model: dict[str, list[dict]]) -> None:
    print()
    print("Summary (median across clips):")
    print(f"{'Model':<8} {'Clips':>5} {'Latency':>10} {'RTF':>7} {'WER':>7}")
    for model, rows in rows_by_model.items():
        lats = [r["latency_s"] for r in rows]
        rtfs = [r["rtf"]       for r in rows]
        wers = [r["wer"] for r in rows if r["wer"] is not None]
        med_lat = statistics.median(lats) * 1000
        med_rtf = statistics.median(rtfs)
        med_wer = fmt_pct(statistics.median(wers)) if wers else "—"
        print(f"{model:<8} {len(rows):>5} {med_lat:>7.0f} ms  {med_rtf:>6.3f}  {med_wer:>7}")


def write_markdown(path: Path, rows_by_model: dict[str, list[dict]],
                   sysinfo: dict) -> None:
    L: list[str] = []
    L.append(f"# Benchmark run — STT v{APP_VERSION}")
    L.append("")
    L.append(f"_{sysinfo['timestamp']}_")
    L.append("")
    L.append("## System")
    L.append("")
    L.append(f"- **OS**: {sysinfo['os']}")
    L.append(f"- **CPU**: {sysinfo['cpu']} · {sysinfo['cores']} cores")
    L.append(f"- **Python**: {sysinfo['python']}")
    L.append(f"- **STT**: v{APP_VERSION}")
    L.append("")

    L.append("## Results (median across clips)")
    L.append("")
    L.append("| Model | Clips | Median latency | Median RTF | Median WER |")
    L.append("|---|---:|---:|---:|---:|")
    for model, rows in rows_by_model.items():
        lats = [r["latency_s"] for r in rows]
        rtfs = [r["rtf"]       for r in rows]
        wers = [r["wer"] for r in rows if r["wer"] is not None]
        med_lat = statistics.median(lats) * 1000
        med_rtf = statistics.median(rtfs)
        med_wer = fmt_pct(statistics.median(wers)) if wers else "—"
        L.append(f"| `{model}` | {len(rows)} | {med_lat:.0f} ms | {med_rtf:.3f} | {med_wer} |")
    L.append("")

    L.append("## Per-clip detail")
    L.append("")
    for model, rows in rows_by_model.items():
        L.append(f"### `{model}`")
        L.append("")
        L.append("| Clip | Audio | Latency | RTF | WER |")
        L.append("|---|---:|---:|---:|---:|")
        for r in rows:
            L.append(
                f"| {r['file']} | "
                f"{r['audio_s']:.2f}s | "
                f"{r['latency_s'] * 1000:.0f} ms | "
                f"{r['rtf']:.3f} | "
                f"{fmt_pct(r['wer'])} |"
            )
        L.append("")

    L.append("---")
    L.append("")
    L.append("_Method: each clip is passed to `faster-whisper` with "
             "`beam_size=1`, `language='en'`, `vad_filter=False`. Latency "
             "is wall-clock from the `transcribe()` call to the last "
             "segment yielded. RTF = latency / audio duration. WER = "
             "Levenshtein(word) / len(reference)._")

    path.write_text("\n".join(L), encoding="utf-8")


# ── Entry point ──────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description="STT transcription benchmark")
    p.add_argument("folder", type=Path,
                   help="Folder containing *.wav reference clips")
    p.add_argument("--model", default="all",
                   help="Model to benchmark (default: all three). "
                        f"One of: {', '.join(MODELS)} or 'all'")
    p.add_argument("--out", type=Path, default=None,
                   help="Write markdown results to this file")
    args = p.parse_args()

    folder: Path = args.folder
    if not folder.exists():
        print(f"Not found: {folder}", file=sys.stderr)
        return 2

    wavs = sorted(folder.glob("*.wav"))
    if not wavs:
        print(f"No .wav files in {folder}", file=sys.stderr)
        return 2

    if args.model == "all":
        to_run = MODELS[:]
    elif args.model in MODELS:
        to_run = [args.model]
    else:
        print(f"Unknown model: {args.model} (allowed: {MODELS} + 'all')",
              file=sys.stderr)
        return 2

    sysinfo = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "os":        platform.platform(),
        "cpu":       (platform.processor() or "unknown").strip(),
        "cores":     os.cpu_count() or 0,
        "python":    sys.version.split()[0],
    }

    print(f"STT benchmark · v{APP_VERSION}")
    print(f"  OS:     {sysinfo['os']}")
    print(f"  CPU:    {sysinfo['cpu']}  ({sysinfo['cores']} cores)")
    print(f"  Python: {sysinfo['python']}")
    print(f"  Clips:  {len(wavs)}  (folder: {folder})")
    print(f"  Models: {', '.join(to_run)}")
    print()

    # Pre-read references
    refs: dict[Path, str | None] = {}
    for wav in wavs:
        txt = wav.with_suffix(".txt")
        refs[wav] = txt.read_text(encoding="utf-8").strip() if txt.exists() else None

    rows_by_model: dict[str, list[dict]] = {}
    for model_name in to_run:
        print(f"Loading '{model_name}' (cpu, int8) …")
        tx = Transcriber(model_name=model_name, device="cpu", compute_type="int8")
        t0 = time.perf_counter()
        tx.load_async().join()
        load_s = time.perf_counter() - t0
        print(f"  loaded + warmed up in {load_s:.2f}s")

        rows: list[dict] = []
        for wav in wavs:
            r = measure_clip(tx, wav, refs[wav])
            r["model"] = model_name
            rows.append(r)
            marker = f"WER={fmt_pct(r['wer']):>6}" if r["wer"] is not None else ""
            print(f"  {wav.name:<24} {r['audio_s']:5.2f}s "
                  f"{r['latency_s'] * 1000:>6.0f} ms "
                  f"RTF={r['rtf']:.3f} {marker}")

        rows_by_model[model_name] = rows
        print()

    print_summary(rows_by_model)

    if args.out:
        write_markdown(args.out, rows_by_model, sysinfo)
        print()
        print(f"Wrote {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
