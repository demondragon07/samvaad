import shutil
import subprocess
from pathlib import Path

import pytest
import requests

import config
from core import transcriber
from utils import audio_processor as audio

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")


def make_tone(path: Path, seconds=62):
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
         "-i", f"sine=frequency=440:duration={seconds}:sample_rate=44100", "-ac", "2", str(path)],
        check=True,
    )
    return path


def probe(path, field):
    return subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", field,
         "-of", "default=nk=1:nw=1", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def test_hindi_chunks_fit_sarvam_limit(tmp_path):
    media = make_tone(tmp_path / "meeting.mp3")
    work = tmp_path / "w"
    work.mkdir()
    chunks = audio.prepare_chunks(media, work, config.HINDI)
    assert len(chunks) == 3
    for chunk in chunks:
        assert float(probe(chunk, "format=duration") or probe(chunk, "stream=duration")) < 30
        assert probe(chunk, "stream=sample_rate") == "16000"
        assert probe(chunk, "stream=channels") == "1"


def test_english_uses_long_chunks(tmp_path):
    media = make_tone(tmp_path / "meeting.wav")
    work = tmp_path / "w"
    work.mkdir()
    assert len(audio.prepare_chunks(media, work, config.ENGLISH)) == 1


def test_corrupt_media_gives_clear_error(tmp_path):
    bad = tmp_path / "bad.mp3"
    bad.write_bytes(b"not audio")
    work = tmp_path / "w"
    work.mkdir()
    with pytest.raises(audio.AudioError):
        audio.prepare_chunks(bad, work, config.ENGLISH)


@pytest.mark.parametrize("url", [
    "https://www.youtube.com/watch?v=xlYJhtL0qbQ&t=2s",
    "https://youtu.be/xlYJhtL0qbQ?si=abc",
    "https://www.youtube.com/shorts/xlYJhtL0qbQ",
    "https://m.youtube.com/live/xlYJhtL0qbQ",
])
def test_video_id_is_stable(url):
    assert audio.is_youtube_url(url)
    assert audio.youtube_video_id(url) == "xlYJhtL0qbQ"


def test_non_youtube_rejected():
    assert not audio.is_youtube_url("https://evil.example/watch?v=abc")
    assert not audio.is_youtube_url("D:/meeting.mp3")


class FakeResponse:
    def __init__(self, status, payload=None):
        self.status_code, self._payload, self.text = status, payload or {}, ""

    def json(self):
        return self._payload


def test_sarvam_request_shape_and_retry(tmp_path, monkeypatch):
    monkeypatch.setenv("SARVAM_API_KEY", "key123")
    chunk = tmp_path / "chunk_00000.wav"
    chunk.write_bytes(b"RIFF")
    calls = []
    replies = [FakeResponse(429), FakeResponse(200, {"transcript": " Hello team. "})]

    def post(url, headers, data, files, timeout):
        calls.append((url, headers, data, files))
        return replies.pop(0)

    monkeypatch.setattr(requests, "post", post)
    assert transcriber.transcribe_with_sarvam(chunk) == "Hello team."
    assert len(calls) == 2
    url, headers, data, _ = calls[0]
    assert url == "https://api.sarvam.ai/speech-to-text"
    assert headers == {"api-subscription-key": "key123"}
    assert data["mode"] == "translate"


def test_failed_run_resumes_without_repeating_chunks(tmp_path, monkeypatch):
    chunks = [tmp_path / f"chunk_{i}.wav" for i in range(3)]
    for chunk in chunks:
        chunk.write_bytes(b"x")
    done, should_fail = [], {"on": True}

    def engine(chunk):
        if chunk.name == "chunk_1.wav" and should_fail["on"]:
            raise transcriber.TranscriptionError("boom")
        done.append(chunk.name)
        return chunk.stem

    monkeypatch.setattr(transcriber, "transcribe_with_sarvam", engine)
    with pytest.raises(transcriber.TranscriptionError):
        transcriber.transcribe_all(chunks, config.HINDI)
    should_fail["on"] = False
    assert transcriber.transcribe_all(chunks, config.HINDI) == "chunk_0 chunk_1 chunk_2"
    assert done == ["chunk_0.wav", "chunk_1.wav", "chunk_2.wav"]
