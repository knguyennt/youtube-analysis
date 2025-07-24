# YouTube Analytics Pipeline Flow

## Architecture Overview

```mermaid
graph TD
    %% Data Sources
    CSV[📊 CSV Files<br/>data/video_id/*.csv]
    
    %% Infrastructure
    subgraph "🐳 Docker Infrastructure"
        ZK[Zookeeper<br/>:2181]
        KAFKA[Kafka<br/>:9092]
        PG[(PostgreSQL<br/>:5432)]
        MB[Metabase<br/>:3000]
        MLF_DOCKER[MLflow Docker<br/>:5001]
    end
    
    %% Processing Components
    subgraph "🐍 Python Components"
        PRODUCER[kafka_producer.py<br/>📤 CSV → Kafka]
        PROCESSOR[stream_processor.py<br/>⚙️ Kafka → PostgreSQL]
        LSTM[lstm_pipeline.py<br/>🧠 LSTM Training & Prediction]
        SCHEDULER[prediction_scheduler.py<br/>⏰ Automated Predictions]
    end
    
    %% Storage & Tracking
    subgraph "💾 Local Storage"
        MODELS[models/<br/>*.h5, *.pkl]
        MLRUNS[mlruns/<br/>MLflow Experiments]
        MLF_LOCAL[MLflow Local<br/>:5002]
    end
    
    %% Database Tables
    subgraph "🗄️ PostgreSQL Tables"
        T1[video_stats_processed<br/>7,708+ records]
        T2[predictions<br/>ML predictions]
        T3[model_metrics<br/>Training metrics]
        T4[hourly_aggregations<br/>Time summaries]
    end
    
    %% Data Flow
    CSV --> PRODUCER
    PRODUCER --> KAFKA
    KAFKA --> PROCESSOR
    PROCESSOR --> PG
    
    %% ML Pipeline Flow
    PG --> LSTM
    LSTM --> MODELS
    LSTM --> MLRUNS
    LSTM --> PG
    
    %% Scheduler Flow
    SCHEDULER -.-> LSTM
    SCHEDULER -.-> PG
    
    %% Storage Details
    PG --> T1
    PG --> T2
    PG --> T3
    PG --> T4
    
    %% Visualization
    PG --> MB
    MLRUNS --> MLF_LOCAL
    
    %% Infrastructure Dependencies
    ZK --> KAFKA
    PG --> MB
    
    %% Styling
    classDef storage fill:#e1f5fe
    classDef processing fill:#f3e5f5
    classDef infrastructure fill:#fff3e0
    classDef output fill:#e8f5e8
    
    class CSV,MODELS,MLRUNS,PG storage
    class PRODUCER,PROCESSOR,LSTM,SCHEDULER processing
    class ZK,KAFKA,MLF_DOCKER infrastructure
    class MB,MLF_LOCAL,T1,T2,T3,T4 output
```

## Detailed Data Flow

```mermaid
sequenceDiagram
    participant User
    participant Producer as kafka_producer.py
    participant Kafka
    participant Processor as stream_processor.py
    participant DB as PostgreSQL
    participant LSTM as lstm_pipeline.py
    participant MLflow as MLflow Local
    
    Note over User,MLflow: 1. Infrastructure Setup
    User->>Producer: docker-compose up -d
    User->>Producer: python src/kafka_producer.py
    
    Note over Producer,Kafka: 2. Data Ingestion
    loop CSV Files
        Producer->>Kafka: Stream CSV records
    end
    
    Note over Kafka,DB: 3. Stream Processing
    User->>Processor: python src/stream_processor.py
    loop Real-time Processing
        Processor->>Kafka: Consume messages
        Processor->>Processor: Calculate growth rates
        Processor->>DB: Store processed data
    end
    
    Note over DB,MLflow: 4. ML Pipeline
    User->>LSTM: python src/lstm_pipeline.py
    LSTM->>DB: Fetch training data (7,708+ records)
    LSTM->>LSTM: Train LSTM models (views/likes/comments)
    LSTM->>MLflow: Log experiments & metrics
    LSTM->>LSTM: Save models (*.h5, *.pkl)
    LSTM->>LSTM: Make predictions
    LSTM->>DB: Store predictions & metrics
    
    Note over User,MLflow: 5. Automated Scheduling (Optional)
    User->>LSTM: python src/prediction_scheduler.py
    loop Every N minutes
        LSTM->>LSTM: Check if models exist
        alt Models missing
            LSTM->>LSTM: Auto-train models
        end
        LSTM->>LSTM: Make predictions
        LSTM->>DB: Store results
    end
```

## Component Responsibilities

```mermaid
mindmap
  root((YouTube Analytics Pipeline))
    Data Ingestion
      CSV Files
        Multiple video datasets
        Time-series data
      Kafka Producer
        Real-time streaming
        Message publishing
    Stream Processing
      Kafka Consumer
        Message consumption
        Real-time processing
      Growth Calculation
        Rate analysis
        Feature engineering
      PostgreSQL Storage
        Processed data
        Normalized schema
    Machine Learning
      LSTM Models
        Views prediction
        Likes prediction
        Comments prediction
      Training Pipeline
        Model training
        Hyperparameter tuning
        Performance evaluation
      Prediction Pipeline
        Real-time inference
        Scheduled predictions
        Auto-retraining
    Monitoring & Visualization
      MLflow Tracking
        Experiment logging
        Model versioning
        Metric tracking
      Metabase Dashboards
        Data visualization
        Business intelligence
        Real-time monitoring
      Database Analytics
        SQL queries
        Performance metrics
        Data quality checks
```

## Service Dependencies

```mermaid
graph LR
    subgraph "Required Services"
        A[Zookeeper] --> B[Kafka]
        C[PostgreSQL] --> D[Metabase]
    end
    
    subgraph "Python Pipeline"
        E[kafka_producer.py] --> B
        B --> F[stream_processor.py]
        F --> C
        C --> G[lstm_pipeline.py]
        G --> H[prediction_scheduler.py]
    end
    
    subgraph "Outputs"
        G --> I[MLflow Local :5002]
        C --> D
        G --> J[Trained Models]
    end
    
    %% Dependencies
    B -.-> F
    C -.-> G
    J -.-> H
    
    classDef required fill:#ffcdd2
    classDef pipeline fill:#c8e6c9
    classDef output fill:#bbdefb
    
    class A,B,C required
    class E,F,G,H pipeline
    class I,D,J output
```

## Quick Commands Reference

| Step | Command                                                        | Purpose                    |
|------|----------------------------------------------------------------|----------------------------|
| 1    | `docker-compose up -d`                                         | Start infrastructure       |
| 2    | `pyenv activate youtube-analysis`                              | Activate Python env        |
| 3    | `python src/kafka_producer.py`                                 | Stream CSV → Kafka         |
| 4    | `python src/stream_processor.py`                               | Process Kafka → PostgreSQL |
| 5    | `python src/lstm_pipeline.py`                                  | Train models & predict     |
| 6    | `python src/prediction_scheduler.py --mode auto --interval 15` | Automated predictions      |

## Access Points

- **Data**: PostgreSQL at `localhost:5432`
- **Dashboards**: Metabase at `http://localhost:3000`
- **ML Experiments**: MLflow at `http://localhost:5002`
- **Message Queue**: Kafka at `localhost:9092`
