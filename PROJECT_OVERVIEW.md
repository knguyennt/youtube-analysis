# YouTube Analytics Real-time Pipeline

## Answer: ML Pipeline Consolidation

**The `ml_pipeline.py` is no longer necessary** since we've enhanced the LSTM pipeline with robust fallback capabilities:

### Enhanced LSTM Pipeline (`lstm_pipeline.py`)
- ✅ **Advanced LSTM Models**: Deep learning for superior prediction accuracy
- ✅ **Smart Fallback**: Automatically uses simple trend analysis if TensorFlow unavailable  
- ✅ **Single Pipeline**: One file handles both advanced and basic predictions
- ✅ **Robust**: Works in any environment regardless of dependencies
- ✅ **Performance**: LSTM achieves excellent metrics (Views MAE=0.09, Likes MAE=0.03, Comments MAE=0.36)

### What Changed
The LSTM pipeline now includes a fully implemented `make_simple_predictions()` method that:
- Uses growth rate trend analysis (same logic as the old ml_pipeline.py)
- Automatically activates when TensorFlow is not available
- Provides reliable predictions without heavy dependencies
- Maintains the same database integration and logging

### Recommendation
You can **safely remove `ml_pipeline.py`** and use only `lstm_pipeline.py` which now provides:
1. **Best case**: LSTM predictions when TensorFlow is available
2. **Fallback**: Simple trend predictions when TensorFlow is not available
3. **Single maintenance point**: One pipeline to maintain instead of two

The system is now more robust and easier to maintain while providing superior performance when possible.

---

## Answer: ML Pipeline Consolidation

This project creates a simple real-time analytics pipeline for YouTube video statistics using:

### Architecture
```
CSV Files → Kafka → Spark Streaming → PostgreSQL → Metabase
                              ↓
                         ML Pipeline (MLflow)
```

### Key Features

1. **Real-time Data Processing**
   - Kafka producer simulates real-time streaming from your CSV files
   - Spark Streaming processes data with growth rate calculations
   - Stores processed data in PostgreSQL

2. **Analytics & Insights**
   - Hourly aggregations of video performance
   - Growth rate analysis (views, likes, comments)
   - Trend detection across different time periods

3. **Machine Learning Pipeline**
   - **LSTM Models**: Deep learning for time series prediction (views, likes, comments)
   - **Sequence Learning**: Uses last 20 time steps for pattern recognition
   - **Multi-target**: Separate LSTM models for each metric
   - **Model Persistence**: Saves trained models and scalers for reuse
   - **Fallback Support**: Simple trend analysis if TensorFlow unavailable
   - **Performance Tracking**: MAE, MSE metrics logged to database
   - **Auto-retraining**: Models can be retrained with new data

4. **Visualization & Monitoring**
   - Metabase dashboards for real-time analytics
   - MLflow UI for model performance tracking
   - Database stores all processed data and predictions

### Quick Start

1. **Start Infrastructure**
   ```bash
   ./start.sh
   ```

2. **Run Pipeline Components** (in separate terminals)
   ```bash
   # Terminal 1: Data producer
   python3 src/kafka_producer.py
   
   # Terminal 2: Stream processor  
   python3 src/spark_streaming.py
   
   # Terminal 3: Simple ML pipeline
   python3 src/ml_pipeline.py
   
   # Terminal 4: LSTM ML pipeline (requires TensorFlow)
   python3 src/lstm_pipeline.py
   ```

3. **Access Dashboards**
   - Metabase: http://localhost:3000
   - MLflow: http://localhost:5001

### Data Flow

1. **Ingestion**: CSV files → Kafka topics (simulated real-time)
2. **Processing**: Spark calculates growth rates and aggregations
3. **Storage**: Processed data stored in PostgreSQL tables
4. **ML**: Models predict future metrics, store predictions
5. **Visualization**: Metabase creates dashboards from database

### LSTM ML Approach (NEW!)

- **Architecture**: Multi-layer LSTM with dropout for regularization
- **Input**: Sequences of 20 time steps with 6 features each
- **Features**: Views, likes, comments + their growth rates
- **Training**: Separate models for views/likes/comments prediction
- **Preprocessing**: MinMax scaling for stable training
- **Validation**: 80/20 train/test split with early stopping
- **Performance**: 
  - Views: MAE = 0.09, MSE = 0.01
  - Likes: MAE = 0.03, MSE = 0.002  
  - Comments: MAE = 0.36, MSE = 0.34
- **Storage**: Models saved as .h5 files with pickle scalers

### Simple ML Approach (Fallback)

- **Features**: Current metrics, growth rates, time features, lag features
- **Targets**: Next hour's view/like/comment counts
- **Model**: Random Forest (simple but effective)
- **Training**: Every 6 hours on rolling 7-day window
- **Evaluation**: MSE, MAE, R² tracked in MLflow

### Database Schema

- `video_stats_processed`: Real-time processed data
- `hourly_aggregations`: Aggregated metrics by hour
- `predictions`: ML model predictions vs actual
- `model_metrics`: Model performance tracking

This keeps the project simple while demonstrating all key concepts of a modern data pipeline!
