import argparse

def process_inference_args():
    parser = argparse.ArgumentParser(description="Run inference on ship detection models")
    
    # Input/Output paths
    parser.add_argument("--input_dir", type=str, required=True, 
                       help="Directory containing input images")
    parser.add_argument("--output_dir", type=str, required=True, 
                       help="Directory to save output predictions")
    
    # Model paths
    parser.add_argument("--classification_model_path", type=str,
                       default="./trained_models/classification_model",
                       help="Path to the trained classification model")
    parser.add_argument("--segmentation_model_path", type=str,
                       default="./trained_models/segmentation_model",
                       help="Path to the trained segmentation model")
    
    # Inference parameters
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for inference")
    parser.add_argument("--confidence_threshold", type=float, default=0.5,
                       help="Confidence threshold for classification")
    parser.add_argument("--segmentation_threshold", type=float, default=0.5,
                       help="Threshold for segmentation mask")
    
    # Image parameters
    parser.add_argument("--img_size", type=int, default=768,
                       help="Size of input images")
    
    return parser.parse_args()
