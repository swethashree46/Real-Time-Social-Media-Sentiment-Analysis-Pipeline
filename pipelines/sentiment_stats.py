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