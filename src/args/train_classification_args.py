import argparse

def process_csv_args():
    parser = argparse.ArgumentParser(description="Process CSV file")
    parser.add_argument("--csv_file_path", type=str, help="Path to the CSV file")
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    
    # parser.add_argument("--output_path", type=str, help="Path to the output CSV file")
    # parser.add_argument("--num_datapoints", type=int, default=None, help="Number of rows to process")
    # parser.add_argument("--ship_noship_ratio", type=float, default=1, help="The ratio of ship to no ship rows in the dataset")
    parser.add_argument("--rand_seed", type=int, default=None, help="Random seed")
    parser.add_argument("--pretrained", type=bool, default=True, help="Use pretrained model")
    parser.add_argument("--tl_learning_rate", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--ft_learning_rate", type=float, default=0.00001, help="Learning rate")
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--tl_epochs", type=int, default=10, help="Transfer learning epochs")
    parser.add_argument("--ft_epochs", type=int, default=10, help="Fine tuning epochs")
    return parser.parse_args()
