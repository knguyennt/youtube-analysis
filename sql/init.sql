-- Initialize database schema
CREATE TABLE IF NOT EXISTS video_stats_processed (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP,
    video_id VARCHAR(20),
    view_count INTEGER,
    like_count INTEGER,
    comment_count INTEGER,
    views_growth_rate FLOAT,
    likes_growth_rate FLOAT,
    comments_growth_rate FLOAT,
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS hourly_aggregations (
    id SERIAL PRIMARY KEY,
    video_id VARCHAR(20),
    hour_start TIMESTAMP,
    avg_view_count FLOAT,
    avg_like_count FLOAT,
    avg_comment_count FLOAT,
    total_views_gained INTEGER,
    total_likes_gained INTEGER,
    total_comments_gained INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS model_metrics (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100),
    run_id VARCHAR(100),
    metric_name VARCHAR(50),
    metric_value FLOAT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    video_id VARCHAR(20),
    prediction_timestamp TIMESTAMP,
    predicted_views INTEGER,
    predicted_likes INTEGER,
    predicted_comments INTEGER,
    actual_views INTEGER,
    actual_likes INTEGER,
    actual_comments INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
-- Create indexes for better performance
CREATE INDEX idx_video_stats_video_id_timestamp ON video_stats_processed(video_id, timestamp);
CREATE INDEX idx_hourly_agg_video_id_hour ON hourly_aggregations(video_id, hour_start);
CREATE INDEX idx_predictions_video_id_timestamp ON predictions(video_id, prediction_timestamp);