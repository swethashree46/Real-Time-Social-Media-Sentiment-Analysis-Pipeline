import pytest


def test_anomaly_log_schema():
    # sample row from anomaly_log
    sample_row = {
        "id": 2,
        "tweet": "@user @user thanks for #lyft credit...",
        "label": 0,
        "ml_sentiment": 0,
        "sentiment_label": "negative",
        "sentiment_score": 0.2,
        "sentiment_category": "positive",
        "log_time": "2025-08-25T10:01:00",
        "description": "Discrepancy between label and sentiment"
    }

    expected_keys = {
        "id", "tweet", "label", "ml_sentiment", "sentiment_label",
        "sentiment_score", "sentiment_category", "log_time", "description"
    }

    assert expected_keys.issubset(sample_row.keys())

def test_send_slack_alert(monkeypatch):
    # mock requests.post to avoid real Slack call
    def fake_post(url, data, headers):
        return type("Response", (), {"status_code": 200})()

    monkeypatch.setattr("requests", "post", fake_post)
    response = send_slack_alert("Test message")
    assert response is None  # function only logs, should not fail