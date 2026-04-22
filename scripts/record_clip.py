#!/usr/bin/env python3
"""
Interactive helper for banking reference clips for the benchmark.

    python scripts/record_clip.py

For each clip:
  1. Type the reference transcript (what you will say).
  2. Press Enter to start recording.
  3. Read the sentence aloud at a natural pace.
  4. Press Enter to stop.
The clip is saved as ``benchmarks/clips/clip<NNN>.wav`` with a matching
``.txt`` file holding the reference.

Tip: aim for 5–30 s of speech per clip. Mix sentence lengths, punctuation,
and at least one with proper nouns / numbers so the benchmark covers more
than short greetings.
"""

from __future__ import annotations

import sys
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16_000
OUT_DIR = Path(__file__).resolve().parent.parent / "benchmarks" / "clips"


def save_wav(path: Path, audio: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    """Write a 1-D float32 [-1, 1] array as 16-bit mono PCM."""
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(pcm.tobytes())


def record_until_enter() -> np.ndarray:
    """Open the default mic, record until the user hits Enter, return audio."""
    frames: list[np.ndarray] = []

    def _cb(indata, frames_n, time_info, status):
        frames.append(indata.copy())

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="float32",
        blocksize=1024, callback=_cb,
    )
    stream.start()
    try:
        input()          # wait for Enter
    finally:
        stream.stop()
        stream.close()

    if not frames:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(frames, axis=0).flatten()


def next_clip_num() -> int:
    existing = list(OUT_DIR.glob("clip*.wav"))
    nums = []
    for f in existing:
        stem = f.stem[4:]
        if stem.isdigit():
            nums.append(int(stem))
    return (max(nums) + 1) if nums else 1


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"STT benchmark — clip recorder")
    print(f"Saving to: {OUT_DIR}")
    print()
    print("Type the sentence you want to say, then follow the prompts.")
    print("Leave the prompt blank to quit.")
    print()

    while True:
        try:
            ref = input("Reference transcript (blank to quit):\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not ref:
            break

        input("[Enter] to start recording, [Enter] to stop …")
        audio = record_until_enter()
        dur = len(audio) / SAMPLE_RATE
        print(f"  captured {dur:.2f}s")

        if dur < 0.5:
            print("  too short, discarded.")
            print()
            continue

        n = next_clip_num()
        wav_path = OUT_DIR / f"clip{n:03d}.wav"
        txt_path = OUT_DIR / f"clip{n:03d}.txt"
        save_wav(wav_path, audio)
        txt_path.write_text(ref + "\n", encoding="utf-8")
        print(f"  saved {wav_path.name} + {txt_path.name}")
        print()

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
