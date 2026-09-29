"""
workloads/contention_generator.py
Generates controlled background contention (CPU-bound stress, I/O stress, memory stress)
to benchmark scheduler behavior under contention.
"""

import time
import os
import argparse
import multiprocessing as mp
import numpy as np

def cpu_worker(duration_s):
    """Saturates CPU with continuous matrix operations."""
    end = time.time() + duration_s
    while time.time() < end:
        A = np.random.rand(500, 500)
        B = np.random.rand(500, 500)
        _ = np.dot(A, B)

def io_worker(duration_s):
    """Saturates disk I/O with continuous read/write cycles."""
    end = time.time() + duration_s
    filename = f"temp_io_{os.getpid()}.bin"
    data = os.urandom(1024 * 1024) # 1 MB
    try:
        while time.time() < end:
            with open(filename, "wb") as f:
                f.write(data)
            with open(filename, "rb") as f:
                _ = f.read()
    finally:
        if os.path.exists(filename):
            os.remove(filename)

def memory_worker(duration_s):
    """Allocates and sweeps through memory to stress memory bandwidth & cache."""
    end = time.time() + duration_s
    size_mb = 128
    block = bytearray(size_mb * 1024 * 1024)
    while time.time() < end:
        for i in range(0, len(block), 4096):
            block[i] = (block[i] + 1) % 256

def start_contention(mode="cpu", workers=2, duration_s=10):
    print("=" * 60)
    print(f"Starting Background Contention: Mode={mode.upper()} | Workers={workers} | Duration={duration_s}s")
    print("=" * 60)

    target_fn = cpu_worker if mode == "cpu" else (io_worker if mode == "io" else memory_worker)
    processes = []
    for _ in range(workers):
        p = mp.Process(target_fn=target_fn, args=(duration_s,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print(f"Contention test finished cleanly.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Background Contention Generator")
    parser.add_argument("--mode", choices=["cpu", "io", "memory"], default="cpu", help="Contention type")
    parser.add_argument("--workers", type=int, default=2, help="Number of concurrent worker processes")
    parser.add_argument("--duration", type=int, default=5, help="Duration in seconds")
    args = parser.parse_args()
    start_contention(mode=args.mode, workers=args.workers, duration_s=args.duration)
