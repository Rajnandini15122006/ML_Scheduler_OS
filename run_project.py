"""
run_project.py
Master Demonstration & Execution Script for MLTRAIN-SCHED
Runs the complete end-to-end pipeline:
  1. Verifies dataset
  2. Trains Random Forest classifier
  3. Executes PyTorch ML workload benchmark
  4. Runs complete scheduler comparison (FCFS, RR, Priority, SJF, ML-Guided)
  5. Runs ablation studies (Section 28)
  6. Launches the interactive Streamlit Dashboard
"""

import os
import sys
import subprocess
import time

def print_banner(text):
    print("\n" + "=" * 70)
    print(f" {text}")
    print("=" * 70)

def main():
    print_banner("MLTRAIN-SCHED: MASTER SYSTEM PIPELINE")
    print("Target Architecture: Linux sched_ext / eBPF + User-Space ML Telemetry")
    print("Execution Environment: High-Fidelity Multi-Level Simulation & Dashboard")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base_dir)

    # 1. Dataset Verification / Generation
    dataset_path = os.path.join("data", "workloads_realistic.csv")
    if not os.path.exists(dataset_path):
        print_banner("STEP 1: Generating Realistic Workload Dataset (1000 Tasks)")
        subprocess.run([sys.executable, "src/workload_generator.py"], check=True)
    else:
        print_banner("STEP 1: Workload Dataset Verified (data/workloads_realistic.csv)")

    # 2. Train ML Classifier
    print_banner("STEP 2: Training Random Forest Workload Classifier")
    subprocess.run([sys.executable, "src/ml_classifier.py"], check=True)

    # 3. Real PyTorch Training Workload
    print_banner("STEP 3: Benchmarking PyTorch Multi-Phase ML Training Workload")
    subprocess.run([sys.executable, "workloads/pytorch_train.py", "--epochs", "2", "--batch-size", "128"], check=True)

    # 4. Run Schedulers Comparison
    print_banner("STEP 4: Running Full Scheduler Comparison Suite")
    subprocess.run([sys.executable, "src/compare_schedulers.py"], check=True)

    # 5. Run Ablations
    print_banner("STEP 5: Running Ablation Studies (A0, A1, A2, A5 vs Full Policy)")
    subprocess.run([sys.executable, "src/ablation_study.py"], check=True)

    # 6. Launch Streamlit Dashboard
    print_banner("STEP 6: Launching Interactive Streamlit Dashboard")
    print("Opening Dashboard at: http://localhost:8501")
    print("Press Ctrl+C in this terminal to stop the dashboard when done.")
    print("=" * 70 + "\n")

    subprocess.run(["streamlit", "run", "dashboard/app.py"])

if __name__ == "__main__":
    main()
