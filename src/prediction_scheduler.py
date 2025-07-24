#!/usr/bin/env python3
"""
Prediction Scheduler for YouTube Analytics Pipeline

This script runs predictions at regular intervals for real-time forecasting.
"""

import sys
import argparse
from lstm_pipeline import YouTubeLSTMPipeline

def main():
    parser = argparse.ArgumentParser(description='Run prediction scheduler')
    parser.add_argument(
        '--interval', 
        type=int, 
        default=15, 
        help='Prediction interval in minutes (default: 15)'
    )
    parser.add_argument(
        '--mode',
        choices=['lstm', 'simple', 'auto'],
        default='auto',
        help='Prediction mode: lstm (LSTM only), simple (fallback only), auto (detect best)'
    )
    
    args = parser.parse_args()
    
    print(f"""
🤖 YouTube Analytics Prediction Scheduler
==========================================
Interval: Every {args.interval} minutes
Mode: {args.mode.upper()}
==========================================

Press Ctrl+C to stop the scheduler
""")
    
    pipeline = YouTubeLSTMPipeline()
    
    try:
        if args.mode == 'simple':
            # Force simple predictions only
            pipeline.TENSORFLOW_AVAILABLE = False
        
        pipeline.run_prediction_scheduler(interval_minutes=args.interval)
    except KeyboardInterrupt:
        print("\n✅ Prediction scheduler stopped gracefully")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
