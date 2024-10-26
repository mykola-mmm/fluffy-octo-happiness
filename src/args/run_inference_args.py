import argparse

def process_inference_args():
    parser = argparse.ArgumentParser(description="Run inference on ship detection models")
    parser.add_argument("--classification_model_path", type=str,
                       default="./trained_models/classification_model",
                       help="Path to the trained classification model")
    parser.add_argument("--segmentation_model_path", type=str,
                       default="./trained_models/segmentation_model",
                       help="Path to the trained segmentation model")
    parser.add_argument("--images_dir", type=str,
                       default="/kaggle/input/airbus-ship-detection/test_v2",
                       help="Path to the images directory")
    parser.add_argument("--output_csv_path", type=str,
                       default="/kaggle/working/submission.csv",
                       help="Path to the output CSV file")
    parser.add_argument("--batch_size", type=int,
                       default=8,
                       help="Batch size for inference")
    return parser.parse_args()
