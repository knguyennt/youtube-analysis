#!/bin/bash

# YouTube Analytics Pipeline Startup Script

echo "Starting YouTube Analytics Pipeline..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Docker is not running. Please start Docker first."
    exit 1
fi

# Start infrastructure
echo "Starting infrastructure (Kafka, PostgreSQL, Metabase, MLflow)..."
docker-compose up -d

# Wait for services to be ready
echo "Waiting for services to start..."
sleep 30

# Check if Kafka is ready
echo "Checking Kafka connectivity..."
python3 -c "
from kafka import KafkaProducer
import time
max_retries = 30
for i in range(max_retries):
    try:
        producer = KafkaProducer(bootstrap_servers=['localhost:9092'])
        producer.close()
        print('Kafka is ready!')
        break
    except Exception as e:
        if i == max_retries - 1:
            print(f'Kafka failed to start: {e}')
            exit(1)
        time.sleep(2)
"

# Install Python dependencies
echo "Installing Python dependencies..."
pip3 install -r requirements.txt

# Create Kafka topic
echo "Creating Kafka topic..."
docker exec -it $(docker ps -q -f name=kafka) kafka-topics --create --topic youtube_stats --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1 --if-not-exists

echo "Infrastructure is ready!"
echo ""
echo "Services available at:"
echo "- Kafka: localhost:9092"
echo "- PostgreSQL: localhost:5432"
echo "- Metabase: http://localhost:3000"
echo "- MLflow: http://localhost:5000"
echo ""
echo "To start the pipeline:"
echo "1. Run data producer: python3 src/kafka_producer.py"
echo "2. Run stream processor: python3 src/spark_streaming.py" 
echo "3. Run ML pipeline: python3 src/ml_pipeline.py"
