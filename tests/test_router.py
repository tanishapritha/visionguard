from visionguard.routing.router import choose_route

def test_high_drift_routes_to_vlm():
    assert choose_route(0.9, 0.1).route == "vlm"

def test_high_uncertainty_routes_to_vlm():
    assert choose_route(0.1, 0.8).route == "vlm"

def test_normal_case_stays_on_cv():
    assert choose_route(0.1, 0.1).route == "cv"
