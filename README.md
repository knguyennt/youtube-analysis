# YouTube Analytics Real-time Pipeline

## Overview
A simple real-time analytics pipeline for YouTube video statistics using Kafka, Spark Streaming, and PostgreSQL.

## Architecture
1. **Data Ingestion**: CSV files → Kafka topics (real-time simulation)
2. **Stream Processing**: Kafka → Stream Processor → PostgreSQL
3. **Analytics**: Growth rate calculations, hourly aggregations
4. **ML Pipeline**: LSTM models for views/likes/comments prediction
5. **Storage**: PostgreSQL for processed data and model metrics
6. **Visualization**: Metabase dashboards + MLflow experiment tracking

## Quick Start
```bash
# 1. Start infrastructure services
docker-compose up -d

# 2. Activate Python environment
pyenv activate youtube-analysis

# 3. Run data producer (CSV → Kafka)
python src/kafka_producer.py

# 4. Run stream processor (Kafka → PostgreSQL)
python src/spark_streaming.py

# 5. Run LSTM ML pipeline (Training & Predictions)
python src/lstm_pipeline.py

# 6. Optional: Run prediction scheduler (includes auto-training if needed)
python src/prediction_scheduler.py --mode auto --interval 300
```

## Training & Prediction Modes

### Manual Training
```bash
python src/lstm_pipeline.py  # Trains models + makes predictions
```

### Automated Predictions with Smart Training
```bash
# Runs predictions every 5 minutes, auto-trains if models missing
python src/prediction_scheduler.py --mode auto --interval 5

# Options:
--mode lstm     # LSTM predictions only
--mode simple   # Simple linear predictions
--mode auto     # Auto-detect best method
--interval 15   # Minutes between predictions (default: 15)
```

**Smart Training Logic**: 
- ✅ **First run**: Automatically trains models if they don't exist
- ✅ **Subsequent runs**: Uses existing models for fast predictions
- ⚠️ **Retraining**: Currently manual (run `lstm_pipeline.py` to retrain)

## Services & Ports
- **PostgreSQL**: `localhost:5432` (user: postgres, password: password)
- **Kafka**: `localhost:9092`
- **Metabase**: `http://localhost:3000`
- **MLflow (Docker)**: `http://localhost:5001`
- **MLflow (Local)**: `http://localhost:5002` (with actual experiments)

## Data Flow
1. **CSV files** in `data/` directory get streamed via Kafka producer
2. **Stream processor** consumes Kafka messages and stores to PostgreSQL
3. **LSTM pipeline** trains models and makes predictions
4. **Metrics** stored in PostgreSQL and tracked in MLflow
5. **Visualizations** available in Metabase dashboards

## Features
- **Real-time Data Processing**: CSV → Kafka → PostgreSQL pipeline
- **Growth Rate Analysis**: Automatic calculation of views/likes/comments growth
- **LSTM Predictions**: Deep learning models for time series forecasting
- **Model Tracking**: MLflow integration for experiment management
- **Interactive Dashboards**: Metabase visualization for data insights
- **Automated Scheduling**: Configurable prediction intervals
- **Scalable Architecture**: Docker-based microservices

## Database Tables
- `video_stats_processed`: Main processed data (7,708+ records)
- `predictions`: ML model predictions with timestamps
- `model_metrics`: Training performance tracking
- `hourly_aggregations`: Time-based statistical summaries

## Project Structure
```
├── data/                    # CSV data files organized by video ID
├── src/
│   ├── kafka_producer.py   # CSV → Kafka streaming
│   ├── stream_processor.py # Kafka → PostgreSQL processing
│   ├── lstm_pipeline.py    # LSTM training & predictions
│   └── prediction_scheduler.py # Automated prediction scheduling
├── sql/
│   └── init.sql            # Database schema
├── models/                 # Trained model artifacts
├── mlruns/                 # Local MLflow experiments
└── docker-compose.yml     # Infrastructure setup
```

## Troubleshooting
- **MLflow empty**: Use local MLflow at port 5002, not Docker version at 5001
- **No data in PostgreSQL**: Check if stream processor is running
- **Python environment**: Ensure `pyenv activate youtube-analysis` is active
- **Port conflicts**: MLflow uses 5001 (Docker) and 5002 (local)

## Next Steps
- View experiments: `http://localhost:5002` (MLflow)
- Access dashboards: `http://localhost:3000` (Metabase)
- Query database: `psql -h localhost -p 5432 -U postgres -d youtube_analytics`
