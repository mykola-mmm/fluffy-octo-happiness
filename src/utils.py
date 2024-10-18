import logging
import os
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Script for training/inferring models for the Airbus Ship Segmentation Challenge")
    parser.add_argument("--dataset_path", type=str, help="Path to the input data")
    parser.add_argument("--log_level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        default="INFO", help="Set the logging level")
    parser.add_argument("--task", type=str, choices=["train_classification", "test_classification", "train_segmentation", "infer", "preprocess"],
                        required=True, help="Choose the task to perform: train_classification, test_classification, train_segmentation, infer, or preprocess data")
    parser.add_argument("--raw_csv_file", type=str, help="Path to the raw CSV file for preprocessing")
    parser.add_argument("--processed_csv_dir", type=str, help="Path to save the processed CSV files")
    parser.add_argument("--classification_model_path", type=str, help="Path to save the classification model and weights")
    parser.add_argument("--num_images", type=int, default=10, help="Number of images to test (default: 10)")

    args = parser.parse_args()
    
    if args.task == "train_classification" and (not args.processed_csv_dir or not args.classification_model_path):
        if not args.processed_csv_dir:
            parser.error("--processed_csv_dir is required when --task is set to train_classification")
        if not args.classification_model_path:
            parser.error("--classification_model_path is required when --task is set to train_classification")
    
    if args.task == "test_classification" and (not args.classification_model_path or not args.dataset_path):
        if not args.classification_model_path:
            parser.error("--classification_model_path is required when --task is set to test_classification")
        if not args.dataset_path:
            parser.error("--dataset_path is required when --task is set to test_classification")

    if args.task == "train_segmentation" and not args.processed_csv_dir:
        parser.error("--processed_csv_dir is required when --task is set to train_segmentation")

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
