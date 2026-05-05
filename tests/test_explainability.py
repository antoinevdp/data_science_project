import pandas as pd

from predictive_maintenance.explainability import summarize_top_features


def test_summarize_top_features_returns_ranked_names() -> None:
    importance_frame = pd.DataFrame(
        {
            "feature": ["rpm", "temperature_motor", "vibration_rms"],
            "importance_mean": [0.04, 0.21, 0.17],
        }
    )

    summary = summarize_top_features(importance_frame, top_n=2)

    assert summary == ["temperature_motor", "vibration_rms"]
