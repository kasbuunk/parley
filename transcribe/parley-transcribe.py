"""parley-transcribe <audio file>: print what is said in it.

parley runs this for a voice message (src/voice.rs). faster-whisper
large-v3-turbo, int8 on the CPU, with its VAD; the same pinned model as
the lab's room-transcribe, loaded offline from PARLEY_WHISPER_MODEL or
~/.local/share/room-audio/hf. The language is detected; anything but
Dutch or English is taken for Dutch, since short Dutch clips are often
mistaken for Afrikaans or German.
"""

import glob
import os
import sys

MODEL_GLOB = os.path.expanduser(
    "~/.local/share/room-audio/hf/hub/"
    "models--mobiuslabsgmbh--faster-whisper-large-v3-turbo/snapshots/*"
)


def model_path():
    if os.environ.get("PARLEY_WHISPER_MODEL"):
        return os.environ["PARLEY_WHISPER_MODEL"]
    found = sorted(glob.glob(MODEL_GLOB))
    if not found:
        sys.exit(f"no whisper model at {MODEL_GLOB} (set PARLEY_WHISPER_MODEL)")
    return found[-1]


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: parley-transcribe <audio file>")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from faster_whisper import WhisperModel

    threads = int(os.environ.get("PARLEY_WHISPER_THREADS", "8"))
    model = WhisperModel(model_path(), device="cpu", compute_type="int8", cpu_threads=threads)
    opts = dict(vad_filter=True, beam_size=5, condition_on_previous_text=False)
    segments, info = model.transcribe(sys.argv[1], **opts)
    if info.language not in ("nl", "en"):
        segments, info = model.transcribe(sys.argv[1], language="nl", **opts)
    print(" ".join(s.text.strip() for s in segments).strip())


if __name__ == "__main__":
    main()
