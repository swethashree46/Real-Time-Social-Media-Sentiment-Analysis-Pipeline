import dlt
from pyspark.sql import functions as F
from pyspark.sql.functions import col
import requests
import json

# ----------------------------
# Slack webhook (Databricks secrets recommended in production)
# ----------------------------
SLACK_WEBHOOK = "https://hooks.slack.com/services/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"

def send_slack_alert(message: str):
    """
    Sends Slack alert via webhook.
    """
    try:
        payload = {"text": message}
        requests.post(SLACK_WEBHOOK, data=json.dumps(payload),
                      headers={'Content-Type': 'application/json'})
    except Exception as e:
        print(f"Slack alert failed: {e}")


# ----------------------------
# DLT Table: anomaly_log
# ----------------------------
@dlt.table(
    name="anomaly_log",
    comment="Logs all discrepancies from Silver layer and triggers Slack alerts (batch-safe)"
)
def anomaly_log():
    # Read Silver layer
    valid_tweets = dlt.read("valid_tweets")

    # Find discrepancies between human label and TextBlob/ML sentiment
    anomalies = (
        valid_tweets
        .filter(col("discrepancy") == True)
        .withColumn("log_time", F.current_timestamp())
        .withColumn("description", F.lit("Discrepancy between label and sentiment"))
        .select(
            "id",
            "tweet",
            "label",
            "ml_sentiment",
            "sentiment_label",
            "sentiment_score",
            "sentiment_category",
            "log_time",
            "description"
        )
    )

    # Batch alert: send Slack notification if anomalies exist
    anomaly_count = anomalies.count()  # safe in batch mode
    if anomaly_count > 0:
        msg = f"Anomaly Detected! {anomaly_count} discrepancies found in sentiment pipeline."
        send_slack_alert(msg)

    return anomalies