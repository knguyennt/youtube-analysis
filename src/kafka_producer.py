import os
import csv
import json
import time
import glob
from kafka import KafkaProducer
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CSVKafkaProducer:
    def __init__(self, kafka_server='localhost:9092', topic='youtube_stats'):
        self.producer = KafkaProducer(
            bootstrap_servers=[kafka_server],
            value_serializer=lambda x: json.dumps(x).encode('utf-8')
        )
        self.topic = topic
        
    def send_csv_data(self, csv_file_path):
        """Send CSV data row by row to Kafka topic"""
        logger.info(f"Processing file: {csv_file_path}")
        
        row_count = 0
        with open(csv_file_path, 'r') as file:
            reader = csv.DictReader(file)
            
            for row in reader:
                message = {
                    'timestamp': row['timestamp'],
                    'video_id': row['video_id'],
                    'viewCount': int(row['viewCount']),
                    'likeCount': int(row['likeCount']),
                    'commentCount': int(row['commentCount']),
                    'file_source': os.path.basename(csv_file_path)
                }
                
                self.producer.send(self.topic, value=message)
                row_count += 1
                
                if row_count % 100 == 0:
                    logger.info(f"Sent {row_count} records...")
                
                time.sleep(0.01)  # Simulate real-time streaming
                
        logger.info(f"Sent {row_count} records from {csv_file_path}")
    
    def simulate_realtime_from_directory(self, data_dir='./data'):
        """Simulate real-time streaming by processing CSV files"""
        csv_files = sorted(glob.glob(f"{data_dir}/**/*.csv", recursive=True))
        
        logger.info(f"Found {len(csv_files)} CSV files to process")
        
        for csv_file in csv_files[:5]:  # Process only first 5 files for testing
            self.send_csv_data(csv_file)
            time.sleep(1)  # Pause between files
            
        self.producer.flush()
        logger.info("Finished processing all files")

if __name__ == "__main__":
    producer = CSVKafkaProducer()
    producer.simulate_realtime_from_directory('/Users/nguyennguyen/workspace/youtube-analysis/data')
