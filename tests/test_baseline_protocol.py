from visionguard.baseline.protocol import Detection, evaluate_detections

def test_detection_evaluation_matches_class_and_iou():
    result = evaluate_detections([Detection(1, 0.9, (0, 0, 10, 10))], [(1, (1, 1, 9, 9))])
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0

def test_wrong_class_is_not_a_match():
    result = evaluate_detections([Detection(2, 0.9, (0, 0, 10, 10))], [(1, (0, 0, 10, 10))])
    assert result["true_positive"] == 0.0
