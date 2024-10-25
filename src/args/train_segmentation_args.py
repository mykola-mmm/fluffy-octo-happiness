import argparse

def process_csv_args():
    parser = argparse.ArgumentParser(description="Process CSV file")
    parser.add_argument("--csv_file_path", type=str, help="Path to the CSV file")
    parser.add_argument("--rand_seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset")
    parser.add_argument("--dropout_rate", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--pretrained", type=bool, default=True, help="Pretrained")
    return parser.parse_args()
