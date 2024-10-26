import argparse

def process_download_args():
    parser = argparse.ArgumentParser(description="Download trained models from Google Drive")
    
    # Model IDs from Google Drive
    parser.add_argument("--classification_model_url", type=str, required=True,
                       help="Google Drive file ID for classification model")
    parser.add_argument("--segmentation_model_url", type=str, required=True,
                       help="Google Drive file ID for segmentation model")
    
    # Output paths
    parser.add_argument("--output_dir", type=str, default="./trained_models",
                       help="Directory to save the downloaded models")
    
    return parser.parse_args()
