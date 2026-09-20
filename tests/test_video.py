from pathlib import Path
from visionguard.data.video import read_video

def test_read_video_rejects_invalid_stride(tmp_path: Path):
    try:
        list(read_video(tmp_path / "missing.mp4", stride=0))
    except ValueError as exc:
        assert "stride" in str(exc)
    else:
        raise AssertionError("expected ValueError")

def test_read_video_requires_openable_file(tmp_path: Path):
    path = tmp_path / "not_a_video.mp4"
    path.write_bytes(b"not video")
    try:
        list(read_video(path))
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("expected FileNotFoundError")
