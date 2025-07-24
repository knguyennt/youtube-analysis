import json
import psycopg2
from kafka import KafkaConsumer
from datetime import datetime
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YouTubeStreamProcessor:
    def __init__(self):
        # Database connection
        self.db_config = {
            'host': 'localhost',
            'database': 'youtube_analytics',
            'user': 'postgres',
            'password': 'password',
            'port': '5432'
        }
        
        # Kafka consumer
        self.consumer = KafkaConsumer(
            'youtube_stats',
            bootstrap_servers=['localhost:9092'],
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            auto_offset_reset='latest',
            group_id='youtube_processor'
        )
        
        # Cache for calculating growth rates
        self.video_cache = {}
    
    def calculate_growth_rates(self, data):
        """Calculate growth rates for views, likes, comments"""
        video_id = data['video_id']
        current_views = data['viewCount']
        current_likes = data['likeCount']
        current_comments = data['commentCount']
        
        if video_id in self.video_cache:
            prev_data = self.video_cache[video_id]
            
            # Calculate growth rates
            views_growth = ((current_views - prev_data['viewCount']) / prev_data['viewCount'] * 100) if prev_data['viewCount'] > 0 else 0
            likes_growth = ((current_likes - prev_data['likeCount']) / prev_data['likeCount'] * 100) if prev_data['likeCount'] > 0 else 0
            comments_growth = ((current_comments - prev_data['commentCount']) / prev_data['commentCount'] * 100) if prev_data['commentCount'] > 0 else 0
        else:
            views_growth = likes_growth = comments_growth = 0.0
        
        # Update cache
        self.video_cache[video_id] = {
            'viewCount': current_views,
            'likeCount': current_likes,
            'commentCount': current_comments
        }
        
        return views_growth, likes_growth, comments_growth
    
    def write_to_postgres(self, processed_data):
        """Write processed data to PostgreSQL"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            insert_query = """
                INSERT INTO video_stats_processed 
                (timestamp, video_id, view_count, like_count, comment_count, 
                 views_growth_rate, likes_growth_rate, comments_growth_rate)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            cursor.execute(insert_query, (
                processed_data['timestamp'],
                processed_data['video_id'],
                processed_data['viewCount'],
                processed_data['likeCount'],
                processed_data['commentCount'],
                processed_data['views_growth_rate'],
                processed_data['likes_growth_rate'],
                processed_data['comments_growth_rate']
            ))
            
            conn.commit()
            cursor.close()
            conn.close()
            
            logger.info(f"Stored data for video {processed_data['video_id']} at {processed_data['timestamp']}")
            
        except Exception as e:
            logger.error(f"Error writing to database: {e}")
    
    def start_streaming(self):
        """Start the streaming process"""
        logger.info("Starting YouTube analytics stream processing...")
        
        processed_count = 0
        for message in self.consumer:
            try:
                data = message.value
                
                # Parse timestamp
                timestamp = datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
                
                # Calculate growth rates
                views_growth, likes_growth, comments_growth = self.calculate_growth_rates(data)
                
                # Prepare processed data
                processed_data = {
                    'timestamp': timestamp,
                    'video_id': data['video_id'],
                    'viewCount': data['viewCount'],
                    'likeCount': data['likeCount'],
                    'commentCount': data['commentCount'],
                    'views_growth_rate': views_growth,
                    'likes_growth_rate': likes_growth,
                    'comments_growth_rate': comments_growth
                }
                
                # Write to database
                self.write_to_postgres(processed_data)
                
                processed_count += 1
                if processed_count % 50 == 0:
                    logger.info(f"Processed {processed_count} messages")
                    
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                continue

if __name__ == "__main__":
    processor = YouTubeStreamProcessor()
    processor.start_streaming()
