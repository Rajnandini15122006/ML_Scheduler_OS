"""
workloads/pytorch_train.py
ML Training Workload Simulation for MLTrain-Sched
Executes multi-phase PyTorch training (DataLoader -> Forward -> Backward -> Optimizer -> Checkpoint)
and measures epoch durations, CPU time, and throughput.
"""

import time
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class TrainingMLP(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=1024, num_classes=10):
        super(TrainingMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)

def run_training(epochs=5, batch_size=128, num_samples=5000, checkpoint_interval=2):
    print("=" * 60)
    print("PyTorch ML Training Workload (MLTrain-Sched Benchmark)")
    print(f"PID: {os.getpid()} | Device: CPU | Epochs: {epochs} | Batch Size: {batch_size}")
    print("=" * 60)

    # 1. Dataset Generation
    X = torch.randn(num_samples, 512)
    y = torch.randint(0, 10, (num_samples,))
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 2. Model & Optimizer
    model = TrainingMLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    total_start = time.time()
    telemetry_logs = []

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        running_loss = 0.0
        total_batches = 0

        # Phase 1: Forward & Backward (Compute Heavy)
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            total_batches += 1

        # Phase 2: Checkpoint Simulation (I/O Heavy)
        if epoch % checkpoint_interval == 0:
            ckpt_path = f"checkpoint_epoch_{epoch}.pt"
            torch.save(model.state_dict(), ckpt_path)
            if os.path.exists(ckpt_path):
                os.remove(ckpt_path)

        epoch_time = time.time() - epoch_start
        samples_per_sec = num_samples / epoch_time
        print(f"Epoch {epoch:02d}/{epochs:02d} | Time: {epoch_time:.3f}s | Throughput: {samples_per_sec:.1f} samples/s | Loss: {running_loss / total_batches:.4f}")
        
        telemetry_logs.append({
            "epoch": epoch,
            "epoch_duration_s": epoch_time,
            "samples_per_sec": samples_per_sec,
            "loss": running_loss / total_batches
        })

    total_time = time.time() - total_start
    print("=" * 60)
    print(f"Training Complete! Total Wall Time: {total_time:.3f}s")
    print(f"Average Throughput: {num_samples * epochs / total_time:.1f} samples/s")
    print("=" * 60)
    return telemetry_logs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch ML Training Workload")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    args = parser.parse_args()
    run_training(epochs=args.epochs, batch_size=args.batch_size)
