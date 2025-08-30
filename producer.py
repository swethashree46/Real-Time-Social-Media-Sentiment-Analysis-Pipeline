import json
import time
import boto3
import os

os.environ["AWS_ACCESS_KEY_ID"] = ""
os.environ["AWS_SECRET_ACCESS_KEY"] = ""

STREAM_NAME = "DatabricksProject"
AWS_REGION = "eu-north-1"

kinesis_client = boto3.client('kinesis', region_name=AWS_REGION)

def read_json_records(file_path):
    with open(file_path, "r") as f:
        for line in f:
            if line.strip():  
                yield json.loads(line)

def send_to_kinesis():
    """Send data to Kinesis stream."""
    for record in read_json_records("D:/ANALYST/DataBricks/P1/SentimentAnalysis/AWS/tweet.json"):
        response = kinesis_client.put_record(
            StreamName=STREAM_NAME,
            Data=json.dumps(record),
            PartitionKey=str(record.get("Date", "default"))
        )
        print(f"Sent record: {record}, Response: {response}")
        time.sleep(0.1)

if __name__ == "__main__":
    send_to_kinesis()
