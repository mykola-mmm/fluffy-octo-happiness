import argparse
import pandas as pd

def process_csv_args():
    parser = argparse.ArgumentParser(description="Process CSV file")
    parser.add_argument("--csv_file_path", type=str, help="Path to the CSV file")
    parser.add_argument("--output_path", type=str, help="Path to the output CSV file")
    parser.add_argument("--num_datapoints", type=int, default=None, help="Number of rows to process")
    parser.add_argument("--ship_noship_ratio", type=float, default=1, help="The ratio of ship to no ship rows in the dataset")
    parser.add_argument("--rand_seed", type=int, default=None, help="Random seed")
    return parser.parse_args()
