import numpy as np
import psycopg2
from datetime import datetime, timedelta
import logging
import json
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
import os

# Try to import MLflow
try:
    import mlflow
    import mlflow.tensorflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logging.warning("MLflow not available. Install with: pip install mlflow")

# Try to import TensorFlow/Keras
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning("TensorFlow not available. Install with: pip install tensorflow")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YouTubeLSTMPipeline:
    def __init__(self):
        self.db_config = {
            'host': 'localhost',
            'database': 'youtube_analytics',
            'user': 'postgres',
            'password': 'password',
            'port': '5432'
        }
        self.scaler = MinMaxScaler()
        self.models = {}
        self.sequence_length = 20  # Use last 20 time steps for prediction
        
        # Create models directory
        os.makedirs('models', exist_ok=True)
    
    def get_training_data(self):
        """Fetch training data from PostgreSQL"""
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        query = """
        SELECT 
            timestamp,
            video_id,
            view_count,
            like_count,
            comment_count,
            views_growth_rate,
            likes_growth_rate,
            comments_growth_rate
        FROM video_stats_processed 
        ORDER BY video_id, timestamp
        """
        
        cursor.execute(query)
        results = cursor.fetchall()
        cursor.close()
        conn.close()
        
        return results
    
    def prepare_lstm_data(self, data, target_column):
        """Prepare data for LSTM training"""
        # Group by video_id
        video_data = {}
        for row in data:
            video_id = row[1]
            if video_id not in video_data:
                video_data[video_id] = []
            video_data[video_id].append(row)
        
        X_sequences = []
        y_values = []
        
        for video_id, video_records in video_data.items():
            if len(video_records) < self.sequence_length + 1:
                continue
                
            # Extract features for this video
            features = []
            for record in video_records:
                # Use view_count, like_count, comment_count, growth_rates as features
                feature_vector = [
                    float(record[2]),  # view_count
                    float(record[3]),  # like_count  
                    float(record[4]),  # comment_count
                    float(record[5]) if record[5] is not None else 0.0,  # views_growth_rate
                    float(record[6]) if record[6] is not None else 0.0,  # likes_growth_rate
                    float(record[7]) if record[7] is not None else 0.0,  # comments_growth_rate
                ]
                features.append(feature_vector)
            
            # Normalize features
            features_array = np.array(features)
            features_normalized = self.scaler.fit_transform(features_array)
            
            # Create sequences
            for i in range(len(features_normalized) - self.sequence_length):
                X_sequences.append(features_normalized[i:i+self.sequence_length])
                
                # Target based on column (0=views, 1=likes, 2=comments)
                target_value = features_normalized[i + self.sequence_length][target_column]
                y_values.append(target_value)
        
        return np.array(X_sequences), np.array(y_values)
    
    def create_lstm_model(self, input_shape):
        """Create LSTM model architecture"""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train_lstm_model(self, target='views'):
        """Train LSTM model for specific target"""
        if not TENSORFLOW_AVAILABLE:
            logger.error("TensorFlow not available. Cannot train LSTM model.")
            return None
            
        logger.info(f"Training LSTM model for {target} prediction...")
        
        # Start MLflow run if available
        if MLFLOW_AVAILABLE:
            # Use local directory for MLflow tracking
            mlflow.set_tracking_uri("file:./mlruns")
            mlflow.set_experiment("YouTube LSTM Models")
        
        # Get training data
        data = self.get_training_data()
        
        if len(data) < 100:
            logger.warning("Not enough data for LSTM training. Need at least 100 records.")
            return None
        
        # Target column mapping
        target_mapping = {'views': 0, 'likes': 1, 'comments': 2}
        target_col = target_mapping[target]
        
        # Prepare LSTM data
        X, y = self.prepare_lstm_data(data, target_col)
        
        if len(X) < 10:
            logger.warning(f"Not enough sequences for {target} training. Need at least 10 sequences.")
            return None
        
        logger.info(f"Prepared {len(X)} sequences for training")
        
        # Split data
        split_index = int(0.8 * len(X))
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        
        if MLFLOW_AVAILABLE:
            with mlflow.start_run(run_name=f"LSTM_{target}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                # Log parameters
                mlflow.log_param("target", target)
                mlflow.log_param("sequence_length", self.sequence_length)
                mlflow.log_param("total_sequences", len(X))
                mlflow.log_param("train_sequences", len(X_train))
                mlflow.log_param("test_sequences", len(X_test))
                mlflow.log_param("input_features", X.shape[2])
                
                # Create and train model
                model = self.create_lstm_model((self.sequence_length, X.shape[2]))
                
                # Train with early stopping
                history = model.fit(
                    X_train, y_train,
                    epochs=50,
                    batch_size=32,
                    validation_data=(X_test, y_test),
                    verbose=0,
                    callbacks=[
                        tf.keras.callbacks.EarlyStopping(
                            monitor='val_loss',
                            patience=10,
                            restore_best_weights=True
                        )
                    ]
                )
                
                # Evaluate model
                y_pred = model.predict(X_test, verbose=0)
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                
                # Log metrics
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("mse", mse)
                mlflow.log_metric("epochs_trained", len(history.history['loss']))
                mlflow.log_metric("final_train_loss", history.history['loss'][-1])
                mlflow.log_metric("final_val_loss", history.history['val_loss'][-1])
                
                # Log model
                mlflow.tensorflow.log_model(model, f"lstm_{target}_model")
                
                logger.info(f"LSTM {target} model - MAE: {mae:.6f}, MSE: {mse:.6f}")
        else:
            # Fallback training without MLflow
            model = self.create_lstm_model((self.sequence_length, X.shape[2]))
            
            history = model.fit(
                X_train, y_train,
                epochs=50,
                batch_size=32,
                validation_data=(X_test, y_test),
                verbose=0,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=10,
                        restore_best_weights=True
                    )
                ]
            )
            
            y_pred = model.predict(X_test, verbose=0)
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            
            logger.info(f"LSTM {target} model - MAE: {mae:.6f}, MSE: {mse:.6f}")
        
        # Save model
        model_path = f'models/lstm_{target}_model.h5'
        model.save(model_path)
        
        # Save scaler
        scaler_path = f'models/lstm_{target}_scaler.pkl'
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Store metrics
        metrics = {
            'mae': mae,
            'mse': mse,
            'training_sequences': len(X_train),
            'epochs_trained': len(history.history['loss'])
        }
        self.store_model_metrics(f'lstm_{target}', metrics)
        
        self.models[target] = model
        return model
    
    def load_lstm_model(self, target):
        """Load trained LSTM model"""
        if not TENSORFLOW_AVAILABLE:
            return None
            
        model_path = f'models/lstm_{target}_model.h5'
        scaler_path = f'models/lstm_{target}_scaler.pkl'
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            model = tf.keras.models.load_model(model_path)
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            return model, scaler
        return None, None
    
    def make_lstm_predictions(self):
        """Make predictions using LSTM models"""
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available. Using fallback prediction method.")
            return self.make_simple_predictions()
            
        logger.info("Making LSTM predictions...")
        
        # Get recent data
        data = self.get_training_data()
        
        if len(data) < self.sequence_length:
            logger.warning("Not enough recent data for LSTM predictions.")
            return
        
        # Group by video_id and get latest sequences
        video_data = {}
        for row in data:
            video_id = row[1]
            if video_id not in video_data:
                video_data[video_id] = []
            video_data[video_id].append(row)
        
        predictions_made = 0
        
        for video_id, video_records in video_data.items():
            if len(video_records) < self.sequence_length:
                continue
                
            # Get last sequence for prediction
            latest_features = []
            for record in video_records[-self.sequence_length:]:
                feature_vector = [
                    float(record[2]),  # view_count
                    float(record[3]),  # like_count
                    float(record[4]),  # comment_count  
                    float(record[5]) if record[5] is not None else 0.0,
                    float(record[6]) if record[6] is not None else 0.0,
                    float(record[7]) if record[7] is not None else 0.0,
                ]
                latest_features.append(feature_vector)
            
            # Normalize features
            features_array = np.array(latest_features)
            features_normalized = self.scaler.fit_transform(features_array)
            
            # Reshape for LSTM input
            X_pred = features_normalized.reshape(1, self.sequence_length, -1)
            
            # Load models and make predictions
            predictions = {}
            for target in ['views', 'likes', 'comments']:
                model, scaler = self.load_lstm_model(target)
                if model is not None:
                    pred_normalized = model.predict(X_pred, verbose=0)[0][0]
                    
                    # Denormalize prediction
                    target_mapping = {'views': 0, 'likes': 1, 'comments': 2}
                    target_col = target_mapping[target]
                    
                    # Create dummy array for inverse transform
                    dummy_array = np.zeros((1, features_array.shape[1]))
                    dummy_array[0, target_col] = pred_normalized
                    pred_denormalized = scaler.inverse_transform(dummy_array)[0, target_col]
                    
                    predictions[target] = max(0, int(pred_denormalized))
                else:
                    # Fallback to current value if model not available
                    current_values = {
                        'views': video_records[-1][2],
                        'likes': video_records[-1][3], 
                        'comments': video_records[-1][4]
                    }
                    predictions[target] = current_values[target]
            
            # Store prediction in database
            self.store_prediction(video_id, predictions)
            predictions_made += 1
            
            logger.info(f"LSTM prediction for {video_id}: Views {predictions['views']}, Likes {predictions['likes']}, Comments {predictions['comments']}")
        
        logger.info(f"Made {predictions_made} LSTM predictions")
    
    def make_simple_predictions(self):
        """Fallback simple prediction method"""
        logger.info("Using simple prediction method...")
        
        # Get recent data
        data = self.get_training_data()
        
        if len(data) == 0:
            logger.warning("No data available for predictions.")
            return
            
        # Group by video_id
        video_data = {}
        for row in data:
            video_id = row[1]
            if video_id not in video_data:
                video_data[video_id] = []
            video_data[video_id].append(row)
        
        predictions_made = 0
        
        for video_id, video_records in video_data.items():
            if len(video_records) < 10:
                continue
                
            # Get last 10 records for trend analysis
            last_10 = video_records[-10:]
            
            # Calculate average growth rates (handle None values)
            views_growth = [row[5] for row in last_10 if row[5] is not None]
            likes_growth = [row[6] for row in last_10 if row[6] is not None]
            comments_growth = [row[7] for row in last_10 if row[7] is not None]
            
            avg_views_growth = sum(views_growth) / len(views_growth) if views_growth else 0
            avg_likes_growth = sum(likes_growth) / len(likes_growth) if likes_growth else 0
            avg_comments_growth = sum(comments_growth) / len(comments_growth) if comments_growth else 0
            
            # Get current values
            current_views = last_10[-1][2]
            current_likes = last_10[-1][3]
            current_comments = last_10[-1][4]
            
            # Predict next hour values based on average growth
            predicted_views = int(current_views * (1 + avg_views_growth / 100))
            predicted_likes = int(current_likes * (1 + avg_likes_growth / 100))
            predicted_comments = int(current_comments * (1 + avg_comments_growth / 100))
            
            predictions = {
                'views': max(0, predicted_views),
                'likes': max(0, predicted_likes),
                'comments': max(0, predicted_comments)
            }
            
            # Store prediction in database
            self.store_prediction(video_id, predictions)
            predictions_made += 1
            
            logger.info(f"Simple prediction for {video_id}: Views {predictions['views']}, Likes {predictions['likes']}, Comments {predictions['comments']}")
        
        logger.info(f"Made {predictions_made} simple predictions")
    
    def store_prediction(self, video_id, predictions):
        """Store prediction in database"""
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        prediction_time = datetime.now() + timedelta(hours=1)
        cursor.execute("""
            INSERT INTO predictions 
            (video_id, prediction_timestamp, predicted_views, predicted_likes, predicted_comments)
            VALUES (%s, %s, %s, %s, %s)
        """, (
            video_id,
            prediction_time,
            predictions['views'],
            predictions['likes'],
            predictions['comments']
        ))
        
        conn.commit()
        cursor.close()
        conn.close()
    
    def store_model_metrics(self, model_name, metrics):
        """Store model metrics in database"""
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        for metric_name, metric_value in metrics.items():
            cursor.execute("""
                INSERT INTO model_metrics (model_name, run_id, metric_name, metric_value)
                VALUES (%s, %s, %s, %s)
            """, (model_name, run_id, metric_name, float(metric_value)))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info(f"Stored LSTM metrics for {model_name}: {metrics}")
    
    def train_all_lstm_models(self):
        """Train LSTM models for all targets"""
        logger.info("Training all LSTM models...")
        
        for target in ['views', 'likes', 'comments']:
            self.train_lstm_model(target)
        
        logger.info("LSTM training completed")
    
    def run_lstm_pipeline(self):
        """Run complete LSTM pipeline"""
        logger.info("Starting LSTM ML pipeline...")
        
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available. Install with: pip install tensorflow")
            logger.info("Proceeding with available functionality...")
        
        # Get data summary
        data = self.get_training_data()
        logger.info(f"Found {len(data)} training records")
        
        if len(data) >= 100:
            if TENSORFLOW_AVAILABLE:
                # Train LSTM models
                self.train_all_lstm_models()
                
                # Make LSTM predictions
                self.make_lstm_predictions()
            else:
                logger.info("Using fallback prediction method")
                self.make_simple_predictions()
        else:
            logger.info("Not enough data for LSTM training. Need at least 100 records.")
        
        logger.info("LSTM ML pipeline completed")
    
    def run_prediction_scheduler(self, interval_minutes=15):
        """Run predictions at regular intervals"""
        import time
        
        logger.info(f"Starting prediction scheduler - predictions every {interval_minutes} minutes")
        
        while True:
            try:
                logger.info("🔄 Scheduled prediction run starting...")
                
                # Get current data
                data = self.get_training_data()
                logger.info(f"Found {len(data)} records for prediction")
                
                if len(data) >= 20:  # Need minimum data for predictions
                    if TENSORFLOW_AVAILABLE:
                        # Check if models exist, if not train them
                        models_exist = all(
                            os.path.exists(f'models/lstm_{target}_model.h5') 
                            for target in ['views', 'likes', 'comments']
                        )
                        
                        if not models_exist:
                            logger.info("Models not found, training first...")
                            self.train_all_lstm_models()
                        
                        # Make LSTM predictions
                        self.make_lstm_predictions()
                    else:
                        # Use fallback predictions
                        self.make_simple_predictions()
                else:
                    logger.warning("Not enough data for predictions")
                
                logger.info(f"✅ Prediction completed. Next run in {interval_minutes} minutes")
                
                # Wait for next interval
                time.sleep(interval_minutes * 60)
                
            except KeyboardInterrupt:
                logger.info("Prediction scheduler stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in prediction scheduler: {e}")
                logger.info(f"Retrying in {interval_minutes} minutes...")
                time.sleep(interval_minutes * 60)

if __name__ == "__main__":
    import sys
    pipeline = YouTubeLSTMPipeline()
    
    # Check for scheduler mode
    if len(sys.argv) > 1 and sys.argv[1] == '--scheduler':
        # Run prediction scheduler
        interval = int(sys.argv[2]) if len(sys.argv) > 2 else 15
        pipeline.run_prediction_scheduler(interval_minutes=interval)
    else:
        # Run one-time training and prediction
        pipeline.run_lstm_pipeline()
