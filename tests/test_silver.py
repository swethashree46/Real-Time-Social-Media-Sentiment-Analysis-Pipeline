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