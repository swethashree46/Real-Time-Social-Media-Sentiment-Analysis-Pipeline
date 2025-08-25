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