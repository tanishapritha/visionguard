import json
from pathlib import Path
from visionguard.data.aicity import AICityScene, object_type_id

def test_object_type_mapping_is_stable():
    assert object_type_id("Person") == 0
    assert object_type_id("Forklift") == 1
    assert object_type_id("unknown") == 99

def test_scene_materializes_camera_annotations(tmp_path: Path):
    scene = tmp_path / "Warehouse_000"
    (scene / "videos").mkdir(parents=True)
    (scene / "videos" / "camera_0001.mp4").write_bytes(b"placeholder")
    gt = {"0": [{"object_type": "Person", "object_id": 7,
                 "2d_bounding_box_visible": {"0001": [10, 20, 30, 40]}}]}
    (scene / "ground_truth.json").write_text(json.dumps(gt), encoding="utf-8")
    manifests = AICityScene(scene, scene / "ground_truth.json").manifests()
    assert len(manifests) == 1
    annotations = manifests[0].load_annotations()
    assert annotations[0][0].class_id == 0
    assert annotations[0][0].bbox_xyxy == (10.0, 20.0, 30.0, 40.0)
