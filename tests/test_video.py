from pathlib import Path
import pytest
from visionguard.data.video import read_video

def test_read_video_rejects_invalid_stride(tmp_path: Path):
    with pytest.raises(ValueError, match="stride"):
        list(read_video(tmp_path / "missing.mp4", stride=0))

def test_read_video_requires_openable_file(tmp_path: Path):
    pytest.importorskip("cv2")
    path = tmp_path / "not_a_video.mp4"
    path.write_bytes(b"not video")
    with pytest.raises(FileNotFoundError):
        list(read_video(path))
