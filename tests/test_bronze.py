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
