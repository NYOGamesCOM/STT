# STT · Benchmarks

This document explains how transcription performance is measured for
STT and records the results of each run. Numbers here are reproducible
by anyone with Python and a microphone — there is nothing magical
going on, no cloud component, no hidden hardware.

## Methodology

Each reference clip is a WAV file accompanied by a `.txt` file that
holds what was actually said. The benchmark:

1. Loads the selected Whisper model via **faster-whisper** (the same
   engine the app uses day-to-day) on **CPU with int8 quantisation**.
   No GPU, no separate "fast mode" — these are the numbers you get
   running the real app.
2. For each clip, calls `model.transcribe(wav, beam_size=1,
   vad_filter=False, language='en', without_timestamps=True)` and
   times the full decode with `time.perf_counter()`.
3. Computes:
   - **Latency** — wall-clock milliseconds from `transcribe()` call
     to the last segment returned.
   - **Real-time factor (RTF)** — `latency / audio_duration`. Lower
     is better. `RTF < 1.0` means "faster than real-time" (the
     model finished before the clip would finish playing back).
   - **Word error rate (WER)** — Levenshtein distance at word
     granularity, divided by the reference length. Lower is better.
     `0%` is perfect, `10%` is roughly "one in ten words misheard".

The model is loaded + warmed up (one dummy transcription) before timing
begins so the first real clip doesn't unfairly absorb CTranslate2's
kernel compilation cost.

## Reproducing a run

```bash
# 1. Bank reference clips (only needed once; your voice, your speaking style)
python scripts/record_clip.py
#    Type a sentence, hit Enter to record, hit Enter to stop.
#    Saves to benchmarks/clips/clip001.wav + clip001.txt, and so on.
#    Aim for 10–15 clips covering a range of lengths and content.

# 2. Run the benchmark
python scripts/benchmark.py benchmarks/clips --out BENCHMARKS-local.md
#    Runs tiny, base and small by default.
#    Use --model base to run just one.
```

The run prints per-clip numbers to stdout and writes a Markdown summary
to `BENCHMARKS-local.md` which you can paste into the **Results** section
below.

## Clip guidance

Good benchmark sets include:

- A few short utterances (2–5 s) — matches the typical push-to-talk
  case (dictate a search query, a chat reply).
- A few medium clips (10–20 s) — matches dictating a paragraph.
- At least one clip with proper nouns, numbers, or punctuation-heavy
  speech — Whisper's weak spots.
- Ideally some variation in background noise and speaking pace.

Speak naturally. Benchmarks recorded in a silent studio at dictation
pace aren't representative of how people actually use the app.

## Results

_This section is populated by running the benchmark on a consistent
machine and pasting the output. Replace the placeholder when you have
your first run._

### v0.3.1 — (placeholder, awaiting first local run)

```
…run benchmarks/clips then paste the summary table here…
```

## Comparing against other apps

The benchmark script measures **our** pipeline end-to-end on the same
hardware. Comparing against a commercial product like WisperFlow is
not directly supported because:

- Their backend is closed-source; we can't run their model with the
  same clips.
- Their client doesn't expose programmatic transcription (it's
  tied to the push-to-talk UI).
- Their numbers depend on cloud availability and plan tier.

What we *can* do is document their **published claims** (from their
website / demos) and our **measured numbers** side by side, with a
clear note that the comparison isn't apples-to-apples. That lives in
the `README.md`'s comparison section, not here.
