# Prediction Trigger Guide

## When Do Predictions Trigger?

### 🔴 Current Default Behavior (Manual)
- **One-time execution**: Only when you manually run the script
- **Command**: `python src/lstm_pipeline.py`
- **Frequency**: Manual trigger only

### 🟢 Enhanced Automatic Scheduling (New Feature)

#### Option 1: Built-in Scheduler
```bash
# Run predictions every 15 minutes (default)
python src/lstm_pipeline.py --scheduler

# Run predictions every 5 minutes
python src/lstm_pipeline.py --scheduler 5

# Run predictions every hour
python src/lstm_pipeline.py --scheduler 60
```

#### Option 2: Dedicated Scheduler Script
```bash
# Default: Every 15 minutes with auto-detection
python src/prediction_scheduler.py

# Custom interval: Every 10 minutes
python src/prediction_scheduler.py --interval 10

# Force simple predictions only
python src/prediction_scheduler.py --mode simple

# Force LSTM predictions only
python src/prediction_scheduler.py --mode lstm
```

## Prediction Logic Flow

### 1. Data Check
- ✅ **Minimum data**: Needs at least 20 records for predictions
- ✅ **LSTM training**: Needs at least 100 records for model training

### 2. Model Selection
- 🧠 **LSTM Models**: Used when TensorFlow available + models exist
- 📊 **Simple Fallback**: Used when TensorFlow unavailable or models missing
- 🔄 **Auto-training**: Trains models if data sufficient but models missing

### 3. Prediction Triggers

| Scenario                   | Trigger Method      | Frequency              |
|----------------------------|---------------------|------------------------|
| **Development/Testing**    | Manual execution    | On-demand              |
| **Real-time Analytics**    | Scheduled execution | Every 15 min (default) |
| **High-frequency Trading** | Scheduled execution | Every 1-5 min          |
| **Hourly Reports**         | Scheduled execution | Every 60 min           |

### 4. Production Deployment Options

#### A. Cron Job (Linux/macOS)
```bash
# Add to crontab for every 15 minutes
*/15 * * * * cd /path/to/youtube-analysis && pyenv activate youtube-analysis && python src/lstm_pipeline.py --scheduler 15
```

#### B. Background Service
```bash
# Run as background service
nohup python src/prediction_scheduler.py --interval 15 > predictions.log 2>&1 &
```

#### C. Docker Deployment
```yaml
# docker-compose.yml addition
prediction-scheduler:
  build: .
  command: python src/prediction_scheduler.py --interval 15
  depends_on:
    - postgres
    - kafka
```

## Monitoring Predictions

### Check Latest Predictions
```sql
SELECT 
    video_id, 
    prediction_timestamp, 
    predicted_views, 
    predicted_likes, 
    predicted_comments 
FROM predictions 
ORDER BY prediction_timestamp DESC 
LIMIT 10;
```

### Monitor Prediction Performance
```sql
SELECT 
    model_name, 
    metric_name, 
    AVG(metric_value) as avg_performance 
FROM model_metrics 
GROUP BY model_name, metric_name 
ORDER BY model_name;
```

## Recommendation for Your Use Case

**For Real-time Analytics Pipeline:**
```bash
# Start the scheduler in background for continuous predictions
python src/prediction_scheduler.py --interval 15 &
```

This will:
- ✅ Run predictions every 15 minutes automatically
- ✅ Use LSTM models when available (high accuracy)
- ✅ Fall back to simple predictions if needed (reliability)
- ✅ Store all predictions in database for analysis
- ✅ Log performance metrics for monitoring
