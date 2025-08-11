#!/usr/bin/env python3
"""
Retrain All FBRef Models with Enhanced Team Strength Features

Systematically retrains all FBRef models using the enhanced training data
with team strength features. Updates each model to use enhanced features
and saves performance comparisons.

Usage:
  python retrain_all_fbref_models.py
"""
import os
import sys
import subprocess
from pathlib import Path

def run_training_script(script_path):
    """Run a training script and capture output."""
    print(f"\n{'='*60}")
    print(f"Training: {script_path}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=script_path.parent,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout per model
        )
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        if result.returncode == 0:
            print(f"✅ Successfully trained: {script_path.name}")
            return True
        else:
            print(f"❌ Failed to train: {script_path.name}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ Timeout training: {script_path.name}")
        return False
    except Exception as e:
        print(f"💥 Error training {script_path.name}: {e}")
        return False

def main():
    """Retrain all FBRef models with enhanced features."""
    
    # Find the models_fbref directory
    script_dir = Path(__file__).parent
    models_fbref_dir = script_dir / "models_fbref"
    
    if not models_fbref_dir.exists():
        print(f"❌ Models directory not found: {models_fbref_dir}")
        return
    
    # Find all training scripts
    training_scripts = []
    for model_dir in models_fbref_dir.iterdir():
        if model_dir.is_dir():
            # Look for training script
            train_script = model_dir / f"{model_dir.name}_model_train.py"
            if train_script.exists():
                training_scripts.append(train_script)
            else:
                # Alternative naming patterns
                for pattern in ["train_*.py", "*_train.py", "train.py"]:
                    matches = list(model_dir.glob(pattern))
                    if matches:
                        training_scripts.extend(matches)
                        break
    
    if not training_scripts:
        print("❌ No training scripts found")
        return
    
    print(f"🚀 Found {len(training_scripts)} training scripts:")
    for script in training_scripts:
        print(f"  - {script.relative_to(script_dir)}")
    
    # Train each model
    results = {}
    for script in training_scripts:
        model_name = script.parent.name
        success = run_training_script(script)
        results[model_name] = success
    
    # Summary
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    
    successful = sum(results.values())
    total = len(results)
    
    print(f"Models trained: {successful}/{total}")
    
    for model_name, success in results.items():
        status = "✅" if success else "❌"
        print(f"  {status} {model_name}")
    
    if successful == total:
        print(f"\n🎉 All models successfully retrained with team strength features!")
    else:
        print(f"\n⚠️ {total - successful} models failed to train")

if __name__ == "__main__":
    main()
