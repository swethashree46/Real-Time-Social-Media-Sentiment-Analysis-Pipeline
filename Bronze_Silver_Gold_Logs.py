#BRONZE PIPELINE ---------------------------------------------------------------
import dlt
from pyspark.sql.functions import col

@dlt.table(
    name="tweet_data",
    comment="Bronze table: raw tweets with ML predictions"
)
def bronze_table():
    # Read raw tweets
    bronze_df = spark.read.json("/Volumes/workspace/default/tweet/")

    # Read ML predictions
    ml_df = spark.read.json("/Volumes/workspace/default/prediction/tweets_ml_predictions.json")

    # Join on 'id' (or the column that links tweets to predictions)
    combined_df = bronze_df.join(ml_df, on="id", how="left")

    # Optional: ensure ML sentiment is a string
    combined_df = combined_df.withColumn("ml_sentiment", col("label").cast("string"))

    return combined_df


#SILVER PIPELINE-----------------------------------------------------------------

import dlt
from pyspark.sql.functions import col, trim, when, udf
from pyspark.sql.types import DoubleType, StringType
from textblob import TextBlob

# ----------------------------
# UDF 1: TextBlob sentiment score
# ----------------------------
def get_textblob_sentiment(text: str) -> float:
    if not text:
        return 0.0
    return float(TextBlob(text).sentiment.polarity)

textblob_udf = udf(get_textblob_sentiment, DoubleType())

# ----------------------------
# UDF 2: Categorize sentiment score
# ----------------------------
def categorize_sentiment(score: float) -> str:
    if score >= 0.6:
        return "strong positive"
    elif score > 0.1:
        return "positive"
    elif score > -0.1:
        return "neutral"
    elif score > -0.6:
        return "negative"
    else:
        return "strong negative"

categorize_udf = udf(categorize_sentiment, StringType())

# ----------------------------
# Read Bronze table function
# ----------------------------
def read_bronze(bronze_table: str):
    return dlt.read(bronze_table).filter(
        col("tweet").isNotNull() & col("user_id").isNotNull()
    )

# ----------------------------
# Silver Table
# ----------------------------
@dlt.table(
    name="valid_tweets",
    comment="Silver table: cleaned tweets with ML predictions from Bronze + TextBlob sentiment"
)
def silver_table():
    bronze_df = read_bronze("tweet_data")

    return (
        bronze_df
        # Human-readable sentiment from label
        .withColumn(
            "sentiment_label",
            when(col("label") == 1, "positive")
            .when(col("label") == 0, "negative")
            .otherwise("neutral")
        )
        # TextBlob numeric sentiment score
        .withColumn("sentiment_score", textblob_udf(col("tweet")))
        # Sentiment category
        .withColumn("sentiment_category", categorize_udf(col("sentiment_score")))
        # Discrepancy between label and ML prediction already in Bronze
        .withColumn("discrepancy",
        when(col("sentiment_label") != col("sentiment_category"), True).otherwise(False))
        # Clean tweet text
        .withColumn("tweet", trim(col("tweet")))
    )




#GOLD PIPELINE-----------------------------------------------------------------


import dlt
from pyspark.sql import functions as F
from pyspark.sql.functions import col
from pyspark.sql.window import Window

# Hourly Sentiment Trend

@dlt.table(
    name="hourly_sentiment_trend",
    comment="Hourly sentiment trend with counts and percentages"
)
def hourly_sentiment_trend():
    silver_df = dlt.read("valid_tweets")
    
    hourly_df = silver_df.withColumn(
        "hour", F.date_trunc("hour", (F.col("timestamp")/1000).cast("timestamp"))
    ).groupBy("hour", "sentiment_label").agg(
        F.count("*").alias("tweet_count")
    )
    
    window_hour = Window.partitionBy("hour")
    hourly_df = hourly_df.withColumn(
        "percentage",
        (F.col("tweet_count") / F.sum("tweet_count").over(window_hour) * 100).cast("double")
    ).orderBy("hour", "sentiment_label")
    
    return hourly_df


# Discrepant Tweets

@dlt.table(
    name="discrepant_tweets",
    comment="Tweets where human label disagrees with TextBlob sentiment"
)
def discrepant_tweets():
    silver_df = dlt.read("valid_tweets")
    
    return silver_df.filter(col("discrepancy") == True).select(
        "id",
        "label",
        "sentiment_label",
        "sentiment_score",
        "ml_sentiment",
        "sentiment_category"
    )

# Top Negative Words

@dlt.table(
    name="negative_word_frequency",
    comment="Top negative words from negative tweets"
)
def negative_word_frequency():
    silver_df = dlt.read("valid_tweets")
    
    negative_words = [
        'bad','attack','lost','hate','#rip','aww','empty','dead',
        '#sad','#anxiety','#depressed','#depression','death','#melancholy'
    ]
    
    return (
        silver_df.filter(col("sentiment_label") == "negative")
        .select(F.explode(F.split(F.lower(col("tweet")), " ")).alias("word"))
        .filter(col("word").isin(negative_words))
        .groupBy("word")
        .agg(F.count("*").alias("freq"))
        .orderBy(F.desc("freq"))
    )


# Top Positive Words

@dlt.table(
    name="positive_word_frequency",
    comment="Top positive words from positive tweets"
)
def positive_word_frequency():
    silver_df = dlt.read("valid_tweets")
    
    positive_words = [
        'good','love','happy','great','fun','amazing','awesome','#joy','#win','#success','#motivation','#healthy','#love'
    ]
    
    return (
        silver_df.filter(col("sentiment_label") == "positive")
        .select(F.explode(F.split(F.lower(col("tweet")), " ")).alias("word"))
        .filter(col("word").isin(positive_words))
        .groupBy("word")
        .agg(F.count("*").alias("freq"))
        .orderBy(F.desc("freq"))
    )

# Most Frequent Words in Neutral Tweets

@dlt.table(
    name="neutral_word_frequency",
    comment="Top words from tweets categorized as neutral"
)
def neutral_word_frequency():
    silver_df = dlt.read("valid_tweets")
    
    return (
        silver_df.filter(col("sentiment_category") == "neutral")
        .select(F.explode(F.split(F.lower(col("tweet")), " ")).alias("word"))
        .groupBy("word")
        .agg(F.count("*").alias("freq"))
        .orderBy(F.desc("freq"))
    )

# Top Hashtags by Sentiment

@dlt.table(
    name="top_hashtags_by_sentiment",
    comment="Top hashtags per sentiment category"
)
def top_hashtags_by_sentiment():
    silver_df = dlt.read("valid_tweets")
    
    # Split tweet into words, keep only hashtags
    hashtags_df = silver_df.withColumn(
        "word", F.explode(F.split(F.col("tweet"), " "))
    ).filter(F.col("word").startswith("#"))
    
    return (
        hashtags_df.groupBy("word", "sentiment_category")
        .agg(F.count("*").alias("count"))
        .orderBy(F.desc("count"))
    )

#  ML Sentiment Accuracy

@dlt.table(
    name="ml_sentiment_accuracy",
    comment="Daily accuracy metrics comparing ML predictions vs ground truth"
)
def ml_sentiment_accuracy():
    silver_df = dlt.read("valid_tweets")
    
    df = silver_df.withColumn(
        "correct_prediction",
        (col("ml_sentiment") == col("label")).cast("integer")
    )
    
    return df.groupBy(F.date_trunc("day", (col("timestamp")/1000).cast("timestamp")).alias("day")).agg(
        F.count("*").alias("total_tweets"),
        F.sum("correct_prediction").alias("correct_predictions"),
        (F.sum("correct_prediction") / F.count("*") * 100).alias("accuracy_percentage")
    )

# ML vs Label Comparison

@dlt.table(
    name="ml_vs_label_comparison",
    comment="Compare ml_sentiment vs label counts for each sentiment category"
)
def ml_vs_label_comparison():
    silver_df = dlt.read("valid_tweets")
    
    return (
        silver_df.groupBy("label", "ml_sentiment")
        .agg(F.count("*").alias("total_tweets"))
        .orderBy("label", "ml_sentiment")
    )


# LOG PIPELINE-----------------------------------------------------

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