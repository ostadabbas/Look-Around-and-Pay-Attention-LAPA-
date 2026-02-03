#!/usr/bin/env python3
"""
Script to run training and triangulation for all TAP3D sequences.

This script automates the process of:
1. Training refinement models for each category and view set
2. Running the LAPA refinement pipeline for each trained model
3. Organizing results by category and view set
"""

import os
import subprocess
import time
import argparse
from tqdm import tqdm

# Define all view sets
VIEW_SETS = {
    'boxes': ['boxes', 'boxes_alt', 'boxes_alt2'],
    'basketball': ['basketball', 'basketball_alt', 'basketball_alt2'],
    'softball': ['softball', 'softball_alt'],
    'tennis': ['tennis', 'tennis_alt'],
    'football': ['football', 'football_alt'],
    'juggle': ['juggle', 'juggle_alt']
}

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run training and triangulation for all TAP3D sequences")
    
    # General parameters
    parser.add_argument("--data_root", type=str, default="./data",
                        help="Root directory containing all TAP3D data categories")
    parser.add_argument("--output_root", type=str, default="./outputs",
                        help="Root directory to save all results")
    parser.add_argument("--calib_file", type=str, default="./data/tap3d_boxes/calibration_161029_sports1.json",
                        help="Path to calibration file (same for all categories)")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of epochs for training each model")
    parser.add_argument("--num_frames", type=int, default=20,
                        help="Number of frames to process per epoch")
    
    # Refinement parameters
    parser.add_argument("--frame_idx", type=int, default=60,
                        help="Frame index to visualize")
    parser.add_argument("--track_history", type=int, default=59,
                        help="Number of frames to show in track history")
    parser.add_argument("--target_size", type=int, default=448,
                        help="Target size for image resizing")
    
    # Selection parameters
    parser.add_argument("--categories", type=str, nargs="+", 
                        choices=list(VIEW_SETS.keys()), default=list(VIEW_SETS.keys()),
                        help="Categories to process. Default is all categories.")
    parser.add_argument("--skip_training", action="store_true",
                        help="Skip training and only run refinement")
    parser.add_argument("--skip_refinement", action="store_true",
                        help="Skip refinement and only run training")
    
    # Attention model parameters
    parser.add_argument("--train_attention", action="store_true",
                        help="Train volumetric attention model before refinement")
    parser.add_argument("--attention_model", action="store_true",
                        help="Use pre-trained attention model during refinement")
    parser.add_argument("--attention_epochs", type=int, default=10,
                        help="Number of epochs for training attention model")
    parser.add_argument("--skip_attention_training", action="store_true",
                        help="Skip attention training even if --train_attention is specified")
    
    return parser.parse_args()

def run_command(command, verbose=True):
    """Run a shell command and print output."""
    if verbose:
        print(f"Running: {command}")
    
    process = subprocess.Popen(
        command, 
        shell=True, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    # Stream the output
    for line in process.stdout:
        if verbose:
            print(line.strip())
    
    # Wait for the process to complete
    process.wait()
    
    return process.returncode

def train_attention(args, category, view_set):
    """Train a volumetric attention model for a specific category and view set."""
    print(f"\n{'='*80}")
    print(f"Training attention model for {category} - {view_set}")
    print(f"{'='*80}")
    
    # Set up paths
    data_dir = os.path.join(args.data_root, f"tap3d_{category}")
    output_dir = os.path.join(args.output_root, f"attention_training/{category}/{view_set}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Build command
    command = (
        f"python train_with_attention.py "
        f"--data_dir {data_dir} "
        f"--calibration_file {args.calib_file} "
        f"--epochs {args.attention_epochs} "
        f"--num_frames {args.num_frames} "
        f"--view_set {view_set} "
        f"--output_dir {output_dir}"
    )
    
    # Run command
    return run_command(command)

def train_model(args, category, view_set):
    """Train a refinement model for a specific category and view set."""
    print(f"\n{'='*80}")
    print(f"Training refinement model for {category} - {view_set}")
    print(f"{'='*80}")
    
    # Set up paths
    data_dir = os.path.join(args.data_root, f"tap3d_{category}")
    output_dir = os.path.join(args.output_root, f"refinement_training/{category}/{view_set}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Build command
    command = (
        f"python train_refinement.py "
        f"--data_dir {data_dir} "
        f"--calibration_file {args.calib_file} "
        f"--epochs {args.epochs} "
        f"--num_frames {args.num_frames} "
        f"--view_set {view_set} "
        f"--output_dir {output_dir}"
    )
    
    # Add attention model path if specified
    if args.attention_model:
        # Determine the attention model path for this category
        attention_model_path = os.path.join(
            args.output_root, 
            f"attention_training/{category}/{view_set}/attention_model_epoch_{args.attention_epochs}.pth"
        )
        if os.path.exists(attention_model_path):
            print(f"Using pre-trained attention model: {attention_model_path}")
            command += f" --attention_checkpoint {attention_model_path}"
        else:
            print(f"Warning: Attention model not found at {attention_model_path}")
            print("Proceeding without attention model.")
    
    # Run command
    return run_command(command)

def run_refinement(args, category, view_set):
    """Run LAPA refinement for a specific category and view set."""
    print(f"\n{'='*80}")
    print(f"Running refinement for {category} - {view_set}")
    print(f"{'='*80}")
    
    # Set up paths
    data_dir = os.path.join(args.data_root, f"tap3d_{category}")
    training_dir = os.path.join(args.output_root, f"refinement_training/{category}/{view_set}")
    output_dir = os.path.join(args.output_root, f"lapa_refinement_results/{category}/{view_set}")
    checkpoint = os.path.join(training_dir, f"refinement_model_epoch_{args.epochs}.pth")
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint):
        print(f"Error: Checkpoint not found at {checkpoint}")
        return 1
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Build command
    command = (
        f"python run_lapa_refinement.py "
        f"--data_dir {data_dir} "
        f"--calib_file {args.calib_file} "
        f"--view_set {view_set} "
        f"--category {category} "
        f"--frame_idx {args.frame_idx} "
        f"--track_history {args.track_history} "
        f"--target_size {args.target_size} "
        f"--checkpoint {checkpoint} "
        f"--output_dir {output_dir}"
    )
    
    # Add attention model path if it exists and is requested
    if args.attention_model:
        attention_model_path = os.path.join(
            args.output_root, 
            f"attention_training/{category}/{view_set}/attention_model_epoch_{args.attention_epochs}.pth"
        )
        if os.path.exists(attention_model_path):
            print(f"Using pre-trained attention model for refinement: {attention_model_path}")
            command += f" --attention_checkpoint {attention_model_path}"
        else:
            print(f"Warning: Attention model not found at {attention_model_path}")
            print("Proceeding with refinement without attention model.")
    
    # Run command
    return run_command(command)

def main():
    """Main function to run training and refinement for all sequences."""
    args = parse_args()
    
    # Process each selected category
    for category in args.categories:
        print(f"\n{'#'*100}")
        print(f"Processing category: {category}")
        print(f"{'#'*100}")
        
        # Process each view set in the category
        for view_set in VIEW_SETS[category]:
            # Step 1: Train attention model if requested
            if args.train_attention and not args.skip_attention_training:
                print(f"Starting Step 1: Training volumetric attention for {category} - {view_set}")
                attention_result = train_attention(args, category, view_set)
                if attention_result != 0:
                    print(f"Error training attention model for {category} - {view_set}")
                    if args.attention_model:
                        print("Warning: Continuing without attention model")
                else:
                    print(f"Successfully trained attention model for {category} - {view_set}")
            
            # Step 2: Train refinement model
            if not args.skip_training:
                print(f"Starting Step 2: Training refinement model for {category} - {view_set}")
                train_result = train_model(args, category, view_set)
                if train_result != 0:
                    print(f"Error training refinement model for {category} - {view_set}")
                    continue
            
            # Step 3: Run refinement pipeline
            if not args.skip_refinement:
                print(f"Starting Step 3: Running refinement pipeline for {category} - {view_set}")
                refine_result = run_refinement(args, category, view_set)
                if refine_result != 0:
                    print(f"Error running refinement for {category} - {view_set}")
                    continue
            
            print(f"Completed all processing for {category} - {view_set}")
    
    print("\nAll processing completed!")

if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
