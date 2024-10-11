import logging
import os
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Script for training/inferring models for the Airbus Ship Segmentation Challenge")
    parser.add_argument("--dataset_path", type=str, help="Path to the input data")
    parser.add_argument("--log_level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        default="INFO", help="Set the logging level")
    parser.add_argument("--task", type=str, choices=["train", "infer", "preprocess"],
                        required=True, help="Choose the task to perform: train, infer, or preprocess data")
    parser.add_argument("--raw_csv_file", type=str, help="Path to the raw CSV file for preprocessing")
    parser.add_argument("--processed_csv_dir", type=str, help="Path to save the processed CSV files")

    args = parser.parse_args()
    
    if args.task == "train" and not args.processed_csv_dir:
        parser.error("--processed_csv_dir is required when --task is set to train")

    if args.task == "preprocess":
        if not args.raw_csv_file:
            parser.error("--raw_csv_file is required when --task is set to preprocess")
        if not args.processed_csv_dir:
            parser.error("--processed_csv_dir is required when --task is set to preprocess")
    
    return args

def setup_logging():
    # Get log level from environment variable, default to 'INFO' if not set
    log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()
    
    # Convert string log level to corresponding logging constant
    numeric_level = getattr(logging, log_level, None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    logging.basicConfig(level=numeric_level, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

def get_log_level():
    return os.environ.get('LOG_LEVEL', 'INFO').upper()