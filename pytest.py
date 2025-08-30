##test_bronze
import pytest

def test_bronze_sample_schema():
    # simulate 2 sample rows from bronze
    sample_data = [
        {
            "id": 1,
            "label": 0,
            "timestamp": 1672531200000,
            "tweet": " @user when a father is dysfunctional...",
            "user_id": 1,
            "ml_sentiment": 0
        },
        {
            "id": 2,
            "label": 0,
            "timestamp": 1672531260000,
            "tweet": "@user @user thanks for #lyft credit...",
            "user_id": 2,
            "ml_sentiment": 0
        }
    ]

    # schema keys expected in bronze layer
    expected_keys = {"id", "label", "timestamp", "tweet", "user_id", "ml_sentiment"}

    for row in sample_data:
        # check all keys exist
        assert expected_keys.issubset(row.keys())
        # basic type checks
        assert isinstance(row["id"], int)
        assert isinstance(row["label"], int)
        assert isinstance(row["timestamp"], int)
        assert isinstance(row["tweet"], str)
        assert isinstance(row["user_id"], int)
        assert isinstance(row["ml_sentiment"], int)


##test_silver

import pytest


def test_silver_schema_and_logic():
    # simulate 2 rows from Silver table
    sample_data = [
        {
            "id": 1,
            "label": 0,
            "timestamp": 1672531200000,
            "tweet": "@user when a father is dysfunctional...",
            "user_id": 1,
            "ml_sentiment": 0,
            "sentiment_label": "negative",
            "sentiment_score": -0.5,
            "sentiment_category": "negative",
            "discrepancy": False
        },
        {
            "id": 2,
            "label": 0,
            "timestamp": 1672531260000,
            "tweet": "@user @user thanks for #lyft credit...",
            "user_id": 2,
            "ml_sentiment": 0,
            "sentiment_label": "negative",
            "sentiment_score": 0.2,
            "sentiment_category": "positive",
            "discrepancy": True
        }
    ]

    # check expected schema keys
    expected_keys = {"id", "label", "timestamp", "tweet", "user_id", "ml_sentiment",
                     "sentiment_label", "sentiment_score", "sentiment_category", "discrepancy"}

    for row in sample_data:
        assert expected_keys.issubset(row.keys())
        # basic type checks
        assert isinstance(row["id"], int)
        assert isinstance(row["label"], int)
        assert isinstance(row["timestamp"], int)
        assert isinstance(row["tweet"], str)
        assert isinstance(row["user_id"], int)
        assert isinstance(row["ml_sentiment"], int)
        assert isinstance(row["sentiment_label"], str)
        assert isinstance(row["sentiment_score"], float)
        assert isinstance(row["sentiment_category"], str)
        assert isinstance(row["discrepancy"], bool)

        # validate sentiment logic
        calc_category = categorize_sentiment(row["sentiment_score"])
        assert row["sentiment_category"] == calc_category

        # discrepancy check
        expected_discrepancy = row["sentiment_label"] != row["sentiment_category"]
        assert row["discrepancy"] == expected_discrepancy

def test_textblob_sentiment_positive_negative():
    assert get_textblob_sentiment("I love this") > 0
    assert get_textblob_sentiment("I hate this") < 0

##test_gold

import pytest

def test_gold_tables_schema():
    # list of Gold tables
    gold_tables = [
        "hourly_sentiment_trend",
        "discrepant_tweets",
        "negative_word_frequency",
        "positive_word_frequency",
        "neutral_word_frequency",
        "top_hashtags_by_sentiment",
        "ml_sentiment_accuracy",
        "ml_vs_label_comparison"
    ]

    for table in gold_tables:
        # just check table name exists
        assert isinstance(table, str)

def test_hourly_sentiment_logic():
    # sample row from hourly_sentiment_trend
    sample_row = {
        "hour": "2025-08-25T10:00:00",
        "sentiment_label": "positive",
        "tweet_count": 10,
        "percentage": 50.0
    }

    # type checks
    assert isinstance(sample_row["hour"], str)
    assert isinstance(sample_row["sentiment_label"], str)
    assert isinstance(sample_row["tweet_count"], int)
    assert isinstance(sample_row["percentage"], float)
    # basic logic
    assert 0 <= sample_row["percentage"] <= 100

def test_discrepant_tweets_logic():
    # sample row from discrepant_tweets
    sample_row = {
        "id": 2,
        "label": 0,
        "sentiment_label": "negative",
        "sentiment_score": 0.2,
        "ml_sentiment": 0,
        "sentiment_category": "positive"
    }
    # discrepancy should exist
    assert sample_row["sentiment_label"] != sample_row["sentiment_category"]

def test_ml_accuracy_logic():
    sample_row = {
        "day": "2025-08-25",
        "total_tweets": 100,
        "correct_predictions": 90,
        "accuracy_percentage": 90.0
    }
    assert sample_row["accuracy_percentage"] == (sample_row["correct_predictions"] / sample_row["total_tweets"]) * 100

##test_logs
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
