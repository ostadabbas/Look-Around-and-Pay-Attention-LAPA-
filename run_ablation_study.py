#!/usr/bin/env python3
"""
Ablation Study for Multi-Camera Multi-Point Tracking System

This script runs a comprehensive ablation study on the tracking system, evaluating:
1. Attention mechanism (with vs. without)
2. Number of cameras (2, 3, 4, 5)
3. Grid size (16 vs. 24)
4. Loss function components
5. Geometric constraint type (epipolar vs. SfM)

Results are saved in CSV format with metrics including:
- Jaccard index (AJ)
- Average position accuracy (< δx avg)
- Occlusion accuracy (OA)
"""

import os
import subprocess
import time
import argparse
import json
import csv
import numpy as np
import random
from tqdm import tqdm
import torch
from pathlib import Path

# Define primary view sets for each category
PRIMARY_VIEW_SETS = {
    'boxes': ['boxes'],
    'basketball': ['basketball'],
    'softball': ['softball'],
    'tennis': ['tennis'],
    'football': ['football'],
    'juggle': ['juggle']
}

# Define camera configurations for ablation
CAMERA_CONFIGS = {
    'boxes': {
        '2cam': ['00_05', '00_23'],
        '3cam': ['00_05', '00_23', '00_16'],
        '4cam': ['00_05', '00_23', '00_16', '00_04'],
        '5cam': ['00_05', '00_23', '00_16', '00_04', '00_03']
    }
}

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run ablation studies for multi-camera multi-point tracking")
    
    # Data and output paths
    parser.add_argument("--data_root", type=str, default="./data", help="Root directory for TAP3D data")
    parser.add_argument("--calib_file", type=str, default="./data/tap3d_boxes/calibration_161029_sports1.json", help="Path to calibration file")
    parser.add_argument("--output_root", type=str, default="./ablation_results", help="Root directory for output results")
    
    # Categories and view sets
    parser.add_argument("--categories", nargs="+", default=["boxes"], choices=["boxes", "basketball", "softball", "tennis", "football", "juggle"], help="Categories to run ablation on")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs for training")
    parser.add_argument("--num_frames", type=int, default=100, help="Number of frames to use for training")
    parser.add_argument("--chunk_size", type=int, default=8, help="Chunk size for training")
    
    # Ablation options
    parser.add_argument("--only", type=str, choices=["camera", "grid", "attention", "geometry", "loss"], 
                        help="Run only this specific ablation type")
    parser.add_argument("--include_loss", action="store_true", 
                        help="Include loss function component ablations")
    parser.add_argument("--resume", action="store_true", 
                        help="Resume from existing results CSV if it exists")
    parser.add_argument("--geometry_types", type=str, nargs="+", default=["epipolar", "sfm"],
                        help="Geometric constraint types to test (epipolar, sfm)")
    parser.add_argument("--use_attention", type=str, nargs="+", default=["with", "without"],
                        help="Attention modes to test (with, without)")
    parser.add_argument("--grid_sizes", type=int, nargs="+", default=[16, 24],
                        help="Grid sizes to test for attention mechanism (only 16 and 24 are supported)")
    
    return parser.parse_args()

def run_command(command, verbose=True):
    """Run a shell command and capture output."""
    if verbose:
        print(f"Running: {command}")
    
    # Set environment variables to avoid MKL threading issues
    env = os.environ.copy()
    env["MKL_THREADING_LAYER"] = "GNU"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    
    # Run the command and capture output
    try:
        # Use subprocess.run for simplicity and reliability
        result = subprocess.run(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            check=False  # Don't raise exception on non-zero exit
        )
        
        output_lines = result.stdout.splitlines()
        
        # Print output if verbose
        if verbose:
            for line in output_lines:
                print(line)
        
        return result.returncode, output_lines
    
    except Exception as e:
        print(f"Error executing command: {e}")
        return 1, [f"Error: {str(e)}"]

def extract_metrics_from_output(output_lines):
    """Extract metrics from command output."""
    metrics = {
        "jaccard_index": None,
        "position_accuracy": None,
        "occlusion_accuracy": None,
        "mpjpe": None,
        "pck": None
    }
    
    for line in output_lines:
        # Match different possible formats of metric output
        if any(x in line for x in ["Jaccard Index:", "Jaccard Index (AJ):", "Jaccard:"]):
            try:
                value = line.split(":")[-1].strip()
                # Remove any % signs and convert to float
                value = value.replace("%", "").strip()
                metrics["jaccard_index"] = float(value)
            except ValueError:
                continue
                
        elif any(x in line for x in ["Position Accuracy:", "Average Position Accuracy:", "Pos Acc:"]):
            try:
                value = line.split(":")[-1].strip()
                value = value.replace("%", "").strip()
                metrics["position_accuracy"] = float(value)
            except ValueError:
                continue
                
        elif any(x in line for x in ["Occlusion Accuracy:", "Occlusion Accuracy (OA):", "Occ Acc:"]):
            try:
                value = line.split(":")[-1].strip()
                value = value.replace("%", "").strip()
                metrics["occlusion_accuracy"] = float(value)
            except ValueError:
                continue
                
        elif any(x in line for x in ["MPJPE:", "Mean Per Joint Position Error:", "Mean Error:"]):
            try:
                value = line.split(":")[-1].strip()
                metrics["mpjpe"] = float(value)
            except ValueError:
                continue
                
        elif any(x in line for x in ["PCK:", "Percentage of Correct Keypoints:", "Correct Keypoints:"]):
            try:
                value = line.split(":")[-1].strip()
                value = value.replace("%", "").strip()
                metrics["pck"] = float(value)
            except ValueError:
                continue
    
    return metrics

def run_attention_training(args, category, view_set, grid_size, output_dir, geometry_type="epipolar"):
    """Train a volumetric attention model with specified grid size and geometry type."""
    print(f"\n{'='*80}")
    print(f"Training attention model for {category} - {view_set} with grid size {grid_size} and {geometry_type} geometry")
    print(f"{'='*80}")
    
    # Set up paths
    data_dir = os.path.join(args.data_root, f"tap3d_{category}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Build command
    command = (
        f"python train_with_attention.py "
        f"--data_dir {data_dir} "
        f"--calibration_file {args.calib_file} "
        f"--epochs {args.epochs} "
        f"--num_frames {args.num_frames} "
        f"--view_set {view_set} "
        f"--output_dir {output_dir} "
        f"--grid_size {grid_size} "
        f"--geometry_type {geometry_type} "
        f"--chunk_size {args.chunk_size}"
    )
    
    # Run command
    return run_command(command)

def run_refinement_training(args, category, view_set, attention_checkpoint=None, loss_weights=None, output_dir=None):
    """Train a refinement model with optional attention and custom loss weights."""
    print(f"\n{'='*80}")
    print(f"Training refinement model for {category} - {view_set}")
    if attention_checkpoint:
        print(f"Using attention checkpoint: {attention_checkpoint}")
    if loss_weights:
        print(f"Using custom loss weights: {loss_weights}")
    print(f"{'='*80}")
    
    # Set up paths
    data_dir = os.path.join(args.data_root, f"tap3d_{category}")
    if output_dir is None:
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
        f"--output_dir {output_dir} "
        f"--chunk_size {args.chunk_size}"
    )
    
    # Add attention model path if specified
    if attention_checkpoint and os.path.exists(attention_checkpoint):
        command += f" --attention_checkpoint {attention_checkpoint}"
    
    # Add custom loss weights if specified
    if loss_weights:
        command += (
            f" --lambda_reconstruction {loss_weights['reconstruction']} "
            f" --lambda_temporal {loss_weights['temporal']} "
            f" --lambda_identity {loss_weights['identity']}"
        )
    
    # Run command
    return run_command(command)

def evaluate_model(args, category, view_set, checkpoint, attention_checkpoint=None, output_dir=None):
    """Evaluate a trained model and extract metrics."""
    print(f"\n{'='*80}")
    print(f"Evaluating model for {category} - {view_set}")
    print(f"{'='*80}")
    
    # Set up paths
    data_dir = os.path.join(args.data_root, f"tap3d_{category}")
    if output_dir is None:
        output_dir = os.path.join(args.output_root, f"evaluation/{category}/{view_set}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Build command
    command = (
        f"python evaluate_ablation.py "
        f"--data_dir {data_dir} "
        f"--calibration_file {args.calib_file} "
        f"--view_set {view_set} "
        f"--checkpoint {checkpoint} "
        f"--output_dir {output_dir} "
        f"--skip_visualization "
        f"--num_frames {args.num_frames}"
    )
    
    # Add attention model path if specified
    if attention_checkpoint and os.path.exists(attention_checkpoint):
        command += f" --attention_checkpoint {attention_checkpoint}"
    
    # Run command
    return_code, output_lines = run_command(command)
    
    # Extract metrics from output
    metrics = extract_metrics_from_output(output_lines)
    
    # If metrics extraction failed, print error and return empty metrics
    if all(value is None for value in metrics.values()):
        print("ERROR: Could not extract metrics from output. Check the evaluation script.")
        print("Output lines:")
        for line in output_lines[-20:]:  # Print the last 20 lines for debugging
            print(f"  {line}")
    
    return metrics

def run_camera_ablation(args, category, view_set):
    """Run ablation study on number of cameras."""
    results = []
    
    for cam_config, camera_list in CAMERA_CONFIGS[category].items():
        print(f"\n{'='*80}")
        print(f"Running camera ablation for {category} with {cam_config}: {camera_list}")
        print(f"{'='*80}")
        
        # Create a custom view set with specific cameras
        custom_view_dir = os.path.join(args.output_root, f"camera_ablation/{category}/{cam_config}")
        os.makedirs(custom_view_dir, exist_ok=True)
        
        # Create a view set configuration file
        view_config = {
            "name": f"{view_set}_{cam_config}",
            "cameras": camera_list
        }
        
        config_path = os.path.join(custom_view_dir, "view_config.json")
        with open(config_path, 'w') as f:
            json.dump(view_config, f, indent=2)
        
        # Train attention model with epipolar geometry (default)
        attention_dir = os.path.join(custom_view_dir, "attention")
        run_attention_training(args, category, view_set, 16, attention_dir, "epipolar")
        attention_checkpoint = os.path.join(attention_dir, f"attention_model_epipolar_epoch_{args.epochs}.pth")
        
        # Train refinement model
        refinement_dir = os.path.join(custom_view_dir, "refinement")
        run_refinement_training(
            args, 
            category, 
            view_set, 
            attention_checkpoint=attention_checkpoint, 
            output_dir=refinement_dir
        )
        refinement_checkpoint = os.path.join(refinement_dir, f"refinement_model_epoch_{args.epochs}.pth")
        
        # Evaluate model
        metrics = evaluate_model(
            args,
            category,
            view_set,
            refinement_checkpoint,
            attention_checkpoint=attention_checkpoint,
            output_dir=os.path.join(custom_view_dir, "evaluation")
        )
        
        # Add results
        results.append({
            "category": category,
            "view_set": view_set,
            "ablation_type": "camera_count",
            "configuration": cam_config,
            "num_cameras": len(camera_list),
            **metrics
        })
    
    return results

def run_grid_size_ablation(args, category, view_set):
    """Run ablation study on grid size for attention mechanism."""
    results = []
    
    # Define grid sizes if not provided in args
    grid_sizes = getattr(args, 'grid_sizes', [16, 24])
    
    for grid_size in grid_sizes:
        print(f"\n{'='*80}")
        print(f"Running grid size ablation for {category} with grid size {grid_size}")
        print(f"{'='*80}")
        
        # Create output directories
        grid_dir = os.path.join(args.output_root, f"grid_ablation/{category}/grid_{grid_size}")
        os.makedirs(grid_dir, exist_ok=True)
        
        # Train attention model with specific grid size (using epipolar geometry by default)
        attention_dir = os.path.join(grid_dir, "attention")
        run_attention_training(args, category, view_set, grid_size, attention_dir, "epipolar")
        attention_checkpoint = os.path.join(attention_dir, f"attention_model_epipolar_epoch_{args.epochs}.pth")
        
        # Train refinement model
        refinement_dir = os.path.join(grid_dir, "refinement")
        run_refinement_training(
            args, 
            category, 
            view_set, 
            attention_checkpoint=attention_checkpoint, 
            output_dir=refinement_dir
        )
        refinement_checkpoint = os.path.join(refinement_dir, f"refinement_model_epoch_{args.epochs}.pth")
        
        # Evaluate model
        metrics = evaluate_model(
            args,
            category,
            view_set,
            refinement_checkpoint,
            attention_checkpoint=attention_checkpoint,
            output_dir=os.path.join(grid_dir, "evaluation")
        )
        
        # Add results
        results.append({
            "category": category,
            "view_set": view_set,
            "ablation_type": "grid_size",
            "configuration": f"grid_{grid_size}",
            "grid_size": grid_size,
            **metrics
        })
    
    return results

def run_attention_ablation(args, category, view_set):
    """Run ablation study on attention mechanism (with vs. without)."""
    results = []
    
    # Define attention modes if not provided in args
    attention_modes = getattr(args, 'use_attention', ["with", "without"])
    
    for attention_mode in attention_modes:
        print(f"\n{'='*80}")
        print(f"Running attention ablation for {category} with attention: {attention_mode}")
        print(f"{'='*80}")
        
        # Create output directories
        attention_ablation_dir = os.path.join(args.output_root, f"attention_ablation/{category}/{attention_mode}")
        os.makedirs(attention_ablation_dir, exist_ok=True)
        
        attention_checkpoint = None
        if attention_mode == "with":
            # Train attention model with epipolar geometry (default)
            attention_dir = os.path.join(attention_ablation_dir, "attention")
            run_attention_training(args, category, view_set, 16, attention_dir, "epipolar")
            attention_checkpoint = os.path.join(attention_dir, f"attention_model_epipolar_epoch_{args.epochs}.pth")
            
            # Train attention model with SfM geometry
            attention_dir_sfm = os.path.join(attention_ablation_dir, "attention_sfm")
            run_attention_training(args, category, view_set, 16, attention_dir_sfm, "sfm")
            attention_checkpoint_sfm = os.path.join(attention_dir_sfm, f"attention_model_sfm_epoch_{args.epochs}.pth")
        
        # Train refinement model
        refinement_dir = os.path.join(attention_ablation_dir, "refinement")
        run_refinement_training(
            args, 
            category, 
            view_set, 
            attention_checkpoint=attention_checkpoint, 
            output_dir=refinement_dir
        )
        refinement_checkpoint = os.path.join(refinement_dir, f"refinement_model_epoch_{args.epochs}.pth")
        
        # Evaluate model
        metrics = evaluate_model(
            args,
            category,
            view_set,
            refinement_checkpoint,
            attention_checkpoint=attention_checkpoint,
            output_dir=os.path.join(attention_ablation_dir, "evaluation")
        )
        
        # Add results
        results.append({
            "category": category,
            "view_set": view_set,
            "ablation_type": "attention",
            "configuration": attention_mode,
            "attention": attention_mode,
            **metrics
        })
    
    return results

def run_loss_ablation(args, category, view_set):
    """Run ablation study on loss function components."""
    results = []
    
    # Define comprehensive loss weight configurations to test various combinations
    loss_configs = [
        # Standard baseline
        {"name": "baseline", "reconstruction": 1.0, "temporal": 0.5, "identity": 1.0},
        
        # Balance reconstruction with other losses based on observed magnitudes
        {"name": "balanced_recon_01", "reconstruction": 0.01, "temporal": 0.5, "identity": 1.0},  # ~1.28 magnitude for recon
        {"name": "balanced_recon_005", "reconstruction": 0.005, "temporal": 0.5, "identity": 1.0},  # ~0.64 magnitude for recon
        
        # Higher temporal weight
        {"name": "high_temporal", "reconstruction": 0.01, "temporal": 2.0, "identity": 1.0},
        
        # Higher identity weight
        {"name": "high_identity", "reconstruction": 0.01, "temporal": 0.5, "identity": 2.0},
        
        # Lower identity weight
        {"name": "low_identity", "reconstruction": 0.01, "temporal": 0.5, "identity": 0.5},
        
        # No temporal - which performed well previously
        {"name": "no_temporal", "reconstruction": 0.01, "temporal": 0.0, "identity": 1.0}
    ]
    
    # First train attention model (shared across loss ablations)
    attention_dir = os.path.join(args.output_root, f"loss_ablation/{category}/attention")
    run_attention_training(args, category, view_set, 16, attention_dir, "epipolar")
    attention_checkpoint = os.path.join(attention_dir, f"attention_model_epipolar_epoch_{args.epochs}.pth")
    
    for config in loss_configs:
        print(f"\n{'='*80}")
        print(f"Running loss ablation for {category} with config: {config['name']}")
        print(f"{'='*80}")
        
        # Create output directories
        loss_dir = os.path.join(args.output_root, f"loss_ablation/{category}/{config['name']}")
        os.makedirs(loss_dir, exist_ok=True)
        
        # Train refinement model with custom loss weights
        refinement_dir = os.path.join(loss_dir, "refinement")
        loss_weights = {
            "reconstruction": config["reconstruction"],
            "temporal": config["temporal"],
            "identity": config["identity"]
        }
        
        run_refinement_training(
            args, 
            category, 
            view_set, 
            attention_checkpoint=attention_checkpoint, 
            loss_weights=loss_weights,
            output_dir=refinement_dir
        )
        refinement_checkpoint = os.path.join(refinement_dir, f"refinement_model_epoch_{args.epochs}.pth")
        
        # Evaluate model
        metrics = evaluate_model(
            args,
            category,
            view_set,
            refinement_checkpoint,
            attention_checkpoint=attention_checkpoint,
            output_dir=os.path.join(loss_dir, "evaluation")
        )
        
        # Add results
        results.append({
            "category": category,
            "view_set": view_set,
            "ablation_type": "loss_weights",
            "configuration": config["name"],
            "lambda_reconstruction": config["reconstruction"],
            "lambda_temporal": config["temporal"],
            "lambda_identity": config["identity"],
            **metrics
        })
    
    return results

def save_results_to_csv(results, output_file):
    """Save ablation results to CSV file."""
    if not results:
        print("No results to save.")
        return
    
    # Get all field names from the results
    fieldnames = set()
    for result in results:
        fieldnames.update(result.keys())
    
    # Sort fieldnames for consistency
    fieldnames = sorted(list(fieldnames))
    
    # Write to CSV
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Results saved to {output_file}")

def run_geometry_ablation(args, category, view_set):
    """Run ablation study on geometric constraint type (epipolar vs. SfM)."""
    results = []
    
    for geometry_type in args.geometry_types:
        print(f"\n{'='*80}")
        print(f"Running geometry ablation for {category} with {geometry_type} geometry")
        print(f"{'='*80}")
        
        # Create output directories
        geometry_dir = os.path.join(args.output_root, f"geometry_ablation/{category}/{geometry_type}")
        os.makedirs(geometry_dir, exist_ok=True)
        
        # Train attention model with specific geometry type
        attention_dir = os.path.join(geometry_dir, "attention")
        run_attention_training(args, category, view_set, 16, attention_dir, geometry_type)
        attention_checkpoint = os.path.join(attention_dir, f"attention_model_{geometry_type}_epoch_{args.epochs}.pth")
        
        # Train refinement model
        refinement_dir = os.path.join(geometry_dir, "refinement")
        run_refinement_training(
            args, 
            category, 
            view_set, 
            attention_checkpoint=attention_checkpoint, 
            output_dir=refinement_dir
        )
        refinement_checkpoint = os.path.join(refinement_dir, f"refinement_model_epoch_{args.epochs}.pth")
        
        # Evaluate model
        metrics = evaluate_model(
            args,
            category,
            view_set,
            refinement_checkpoint,
            attention_checkpoint=attention_checkpoint,
            output_dir=os.path.join(geometry_dir, "evaluation")
        )
        
        # Add results
        results.append({
            "category": category,
            "view_set": view_set,
            "ablation_type": "geometry_type",
            "configuration": geometry_type,
            "geometry_type": geometry_type,
            **metrics
        })
    
    return results

def main():
    """Main function to run the ablation study."""
    args = parse_args()
    
    # Create output root directory
    os.makedirs(args.output_root, exist_ok=True)
    
    # Initialize results list
    all_results = []
    
    # Check if we should resume from existing results
    results_file = os.path.join(args.output_root, "ablation_results.csv")
    if args.resume and os.path.exists(results_file):
        try:
            with open(results_file, 'r') as f:
                reader = csv.DictReader(f)
                all_results = list(reader)
            print(f"Resuming from existing results with {len(all_results)} entries")
        except Exception as e:
            print(f"Error loading existing results: {e}")
            all_results = []
    
    # Process each category
    for category in args.categories:
        # Check if this category has data
        category_data_dir = os.path.join(args.data_root, f"tap3d_{category}")
        if not os.path.exists(category_data_dir):
            print(f"\nWARNING: Data directory not found for category '{category}': {category_data_dir}")
            print(f"Skipping category '{category}'")
            continue
        
        # Get the primary view set for this category
        if category in PRIMARY_VIEW_SETS:
            view_sets = PRIMARY_VIEW_SETS[category]
        else:
            print(f"\nWARNING: No primary view set defined for category '{category}'")
            print(f"Using default view set name '{category}'")
            view_sets = [category]
        
        # Process each view set
        for view_set in view_sets:
            print(f"\n{'='*80}")
            print(f"Running ablation studies for category: {category}, view set: {view_set}")
            print(f"{'='*80}")
            
            try:
                # Run camera ablation
                if (args.only is None or args.only == "camera") and category in CAMERA_CONFIGS:
                    print(f"\nRunning camera ablation for {category} - {view_set}")
                    camera_results = run_camera_ablation(args, category, view_set)
                    all_results.extend(camera_results)
                    # Save intermediate results
                    save_results_to_csv(all_results, results_file)
                
                # Run grid size ablation
                if args.only is None or args.only == "grid":
                    print(f"\nRunning grid size ablation for {category} - {view_set}")
                    grid_results = run_grid_size_ablation(args, category, view_set)
                    all_results.extend(grid_results)
                    # Save intermediate results
                    save_results_to_csv(all_results, results_file)
                
                # Run attention ablation
                if args.only is None or args.only == "attention":
                    print(f"\nRunning attention ablation for {category} - {view_set}")
                    attention_results = run_attention_ablation(args, category, view_set)
                    all_results.extend(attention_results)
                    # Save intermediate results
                    save_results_to_csv(all_results, results_file)
                
                # Run geometry type ablation
                if args.only is None or args.only == "geometry":
                    print(f"\nRunning geometry ablation for {category} - {view_set}")
                    geometry_results = run_geometry_ablation(args, category, view_set)
                    all_results.extend(geometry_results)
                    # Save intermediate results
                    save_results_to_csv(all_results, results_file)
                
                # Run loss ablation
                if args.only is None or args.only == "loss":
                    print(f"\nRunning loss ablation for {category} - {view_set}")
                    loss_results = run_loss_ablation(args, category, view_set)
                    all_results.extend(loss_results)
                    # Save intermediate results
                    save_results_to_csv(all_results, results_file)
                    
            except Exception as e:
                print(f"\nERROR running ablation for {category} - {view_set}: {e}")
                print("Continuing with next view set/category...")
    
    # Final save of all results to CSV
    save_results_to_csv(all_results, results_file)
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"Ablation study completed. Results saved to {results_file}")
    print(f"{'='*80}")

if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Total runtime: {int(hours)}h {int(minutes)}m {int(seconds)}s")
