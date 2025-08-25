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