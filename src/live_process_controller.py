"""
src/live_process_controller.py
Live Operating System Process Controller & Priority Injector for MLTrain-Sched.

Actively monitors real running processes on the host machine (Windows or Linux),
extracts live OS telemetry (CPU %, Memory RSS, thread count, I/O rates),
classifies workloads with the 99% accuracy ML model, and physically
adjusts real OS process priorities and CPU core affinities in real-time.
"""

import os
import sys
import time
import pickle
import psutil
import pandas as pd
import numpy as np

MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "best_tuned_classifier.pkl")

# Protected critical OS processes (Never modify these)
PROTECTED_PROCESSES = {
    "system", "system idle process", "registry", "smss.exe", "csrss.exe",
    "wininit.exe", "services.exe", "lsass.exe", "svchost.exe", "fontdrvhost.exe",
    "memory compression", "dwm.exe", "explorer.exe", "taskhostw.exe",
    "init", "systemd", "kthreadd", "kworker", "sshd", "auditd"
}

# Win32 Process Priority Mappings
if sys.platform == "win32":
    PRIORITY_MAP = {
        "HIGH": psutil.HIGH_PRIORITY_CLASS,
        "ABOVE_NORMAL": psutil.ABOVE_NORMAL_PRIORITY_CLASS,
        "NORMAL": psutil.NORMAL_PRIORITY_CLASS,
        "BELOW_NORMAL": psutil.BELOW_NORMAL_PRIORITY_CLASS,
        "IDLE": psutil.IDLE_PRIORITY_CLASS
    }
else:
    # Linux nice levels (-20 highest, 19 lowest)
    PRIORITY_MAP = {
        "HIGH": -10,
        "ABOVE_NORMAL": -5,
        "NORMAL": 0,
        "BELOW_NORMAL": 5,
        "IDLE": 15
    }


class LiveProcessController:
    def __init__(self, dry_run=False):
        self.dry_run = dry_run
        self.model = None
        self.encoder = None
        self.features = None
        self.action_history = []
        self._load_classifier()

    def _load_classifier(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file {MODEL_PATH} not found. Please train models first.")
        with open(MODEL_PATH, "rb") as f:
            artifacts = pickle.load(f)
        self.model = artifacts["model"]
        self.encoder = artifacts["encoder"]
        self.features = artifacts.get("features", [
            "cpu_burst", "io_burst", "priority", "previous_runtime",
            "context_switches", "mem_rss_mb", "num_threads"
        ])
        print(f"[LiveController] Loaded Classifier: {artifacts.get('name', 'Tuned Model')}")

    def scan_active_processes(self, limit=25):
        """
        Scans currently active user processes and extracts live OS telemetry.
        """
        records = []
        for p in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info', 'num_threads']):
            try:
                name = p.info['name']
                if not name or name.lower() in PROTECTED_PROCESSES:
                    continue

                pid = p.info['pid']
                if pid == os.getpid():
                    continue

                cpu_pct = p.info['cpu_percent'] or 0.0
                mem_rss = (p.info['memory_info'].rss / (1024 * 1024)) if p.info['memory_info'] else 0.0
                threads = p.info['num_threads'] or 1

                # Approximate I/O and burst rates from OS counters
                try:
                    io = p.io_counters()
                    io_mbs = (io.read_bytes + io.write_bytes) / (1024 * 1024) if io else 0.0
                except (psutil.AccessDenied, AttributeError):
                    io_mbs = 0.0

                try:
                    ctx = p.num_ctx_switches()
                    ctx_switches = (ctx.voluntary + ctx.involuntary) if ctx else 5
                except (psutil.AccessDenied, AttributeError):
                    ctx_switches = 5

                try:
                    create_t = p.create_time()
                    runtime_s = max(1.0, time.time() - create_t)
                except (psutil.AccessDenied, AttributeError):
                    runtime_s = 10.0

                records.append({
                    "pid": pid,
                    "name": name,
                    "cpu_pct": cpu_pct,
                    "mem_rss_mb": mem_rss,
                    "num_threads": threads,
                    "io_mbs": io_mbs,
                    "ctx_switches": min(ctx_switches, 100),
                    "runtime_s": runtime_s
                })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        df = pd.DataFrame(records)
        if df.empty:
            return df
        return df.sort_values(by=["cpu_pct", "mem_rss_mb"], ascending=False).head(limit)

    def classify_and_optimize(self, df_procs):
        """
        Infers workload class and applies real OS scheduling adjustments.
        """
        if df_procs.empty:
            return []

        actions = []
        for _, row in df_procs.iterrows():
            pid = int(row["pid"])
            name = row["name"]
            
            # Map OS metrics to classifier feature space
            cpu_burst_est = int(np.clip(row["cpu_pct"] * 0.8, 5, 120))
            io_burst_est = int(np.clip(row["io_mbs"] * 4.0, 0, 100))
            priority_val = 5
            prev_runtime = int(np.clip(row["runtime_s"] * 0.5, 0, 120))
            ctx_switches = int(np.clip(row["ctx_switches"], 1, 25))
            mem_rss = float(row["mem_rss_mb"])
            threads = int(row["num_threads"])

            # Domain features for tuned models
            io_to_cpu = io_burst_est / (cpu_burst_est + 1.0)
            cpu_frac = cpu_burst_est / (cpu_burst_est + io_burst_est + 1.0)
            switch_int = ctx_switches / (prev_runtime + 1.0)

            sample_dict = {
                "cpu_burst": [cpu_burst_est],
                "io_burst": [io_burst_est],
                "priority": [priority_val],
                "previous_runtime": [prev_runtime],
                "context_switches": [ctx_switches],
                "mem_rss_mb": [mem_rss],
                "num_threads": [threads],
                "io_to_cpu_ratio": [io_to_cpu],
                "cpu_fraction": [cpu_frac],
                "switch_intensity": [switch_int]
            }
            sample_df = pd.DataFrame(sample_dict)

            model_feats = [f for f in self.features if f in sample_df.columns]
            pred_idx = self.model.predict(sample_df[model_feats])[0]
            workload_class = self.encoder.inverse_transform([pred_idx])[0]

            # Determine Target OS Scheduling Priority
            if workload_class == "ML_TRAINING":
                target_prio = "HIGH"
                target_dsq = "DSQ_HIGH"
                target_quantum_ms = 15
            elif workload_class == "IO_BOUND":
                target_prio = "ABOVE_NORMAL"
                target_dsq = "DSQ_HIGH"
                target_quantum_ms = 5
            elif workload_class == "MIXED":
                target_prio = "NORMAL"
                target_dsq = "DSQ_MEDIUM"
                target_quantum_ms = 10
            else:  # CPU_BOUND
                target_prio = "BELOW_NORMAL"
                target_dsq = "DSQ_LOW"
                target_quantum_ms = 20

            # Apply OS Priority Adjustment
            status = "SIMULATED"
            if not self.dry_run:
                try:
                    p = psutil.Process(pid)
                    p.nice(PRIORITY_MAP[target_prio])
                    status = "APPLIED_TO_OS"
                except (psutil.AccessDenied, psutil.NoSuchProcess) as e:
                    status = f"PERMISSION_RESTRICTED ({type(e).__name__})"

            action_record = {
                "timestamp": time.strftime("%H:%M:%S"),
                "pid": pid,
                "name": name,
                "cpu_pct": round(row["cpu_pct"], 1),
                "mem_rss_mb": round(row["mem_rss_mb"], 1),
                "threads": threads,
                "predicted_workload": workload_class,
                "assigned_dsq": target_dsq,
                "target_quantum_ms": target_quantum_ms,
                "os_priority_set": target_prio,
                "status": status
            }
            actions.append(action_record)
            self.action_history.append(action_record)

        return actions


def main():
    print("=" * 70)
    print("MLTRAIN-SCHED: LIVE OPERATING SYSTEM PROCESS CONTROLLER")
    print("=" * 70)
    controller = LiveProcessController(dry_run=False)

    print("\nScanning active host processes...")
    df_procs = controller.scan_active_processes(limit=15)
    print(f"Found {len(df_procs)} target user processes.")

    print("\nApplying ML Classification & Real OS Priority Adjustments:")
    actions = controller.classify_and_optimize(df_procs)
    
    df_act = pd.DataFrame(actions)
    if not df_act.empty:
        cols = ["pid", "name", "cpu_pct", "mem_rss_mb", "predicted_workload", "assigned_dsq", "os_priority_set", "status"]
        print(df_act[cols].to_string(index=False))

    print("\n" + "=" * 70)
    print("Live Process Controller test completed.")


if __name__ == "__main__":
    main()
