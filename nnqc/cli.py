"""
Command Line Interface for nnQC package
"""

import argparse
import json
import sys
from pathlib import Path

def train_autoencoder():
    """CLI entry point for training autoencoder"""
    try:
        from .training.train_autoencoder import main as train_ae_main
        train_ae_main()
    except ImportError:
        print("Error: train_autoencoder.py not found. Please ensure the file exists in nnqc/training/")
        sys.exit(1)

def train_diffusion():
    """CLI entry point for training diffusion model"""
    try:
        from .training.train_diffusion import main as train_diff_main
        train_diff_main()
    except ImportError:
        print("Error: Could not import train_diffusion module")
        sys.exit(1)

def run_inference():
    """CLI entry point for running inference"""
    from .inference.inference import main as inference_main
    inference_main()

def evaluate_validation():
    """CLI entry point for validation evaluation"""
    from .inference.inference import main as eval_main
    eval_main()

def main():
    """Main CLI entry point with subcommands"""
    parser = argparse.ArgumentParser(
        description="nnQC: Neural Network Quality Control for Medical Segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  nnqc train-ae -c config/config_train_32g.json
  nnqc train-diffusion -c config/config_train_32g.json -g 2
  nnqc inference -c config/config_train_32g.json
  nnqc evaluate -c config/config_train_32g.json
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train autoencoder
    ae_parser = subparsers.add_parser('train-ae', help='Train autoencoder model')
    ae_parser.add_argument('-c', '--config-file', default='./config/config_train_32g.json',
                          help='Config JSON file')
    ae_parser.add_argument('-e', '--environment-file', default='./config/environment.json',
                          help='Environment JSON file')
    ae_parser.add_argument('-g', '--gpus', default=1, type=int, help='Number of GPUs')
    
    # Train diffusion
    diff_parser = subparsers.add_parser('train-diffusion', help='Train diffusion model')
    diff_parser.add_argument('-c', '--config-file', default='./config/config_train_32g.json',
                            help='Config JSON file')
    diff_parser.add_argument('-e', '--environment-file', default='./config/environment.json',
                            help='Environment JSON file')
    diff_parser.add_argument('-g', '--gpus', default=1, type=int, help='Number of GPUs')
    
    # Inference
    inf_parser = subparsers.add_parser('inference', help='Run inference/evaluation')
    inf_parser.add_argument('-c', '--config-file', default='./config/config_train_32g.json',
                           help='Config JSON file')
    inf_parser.add_argument('-e', '--environment-file', default='./config/environment.json',
                           help='Environment JSON file')
    inf_parser.add_argument('-n', '--num', type=int, default=1, help='Number of generated images')
    
    # Evaluate
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate validation set')
    eval_parser.add_argument('-c', '--config-file', default='./config/config_train_32g.json',
                            help='Config JSON file')
    eval_parser.add_argument('-e', '--environment-file', default='./config/environment.json',
                            help='Environment JSON file')
    
    args = parser.parse_args()
    
    if args.command == 'train-ae':
        train_autoencoder()
    elif args.command == 'train-diffusion':
        train_diffusion()
    elif args.command == 'inference':
        run_inference()
    elif args.command == 'evaluate':
        evaluate_validation()
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 