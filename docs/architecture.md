# MLTrain-Sched: System Architecture & Technical Specification

## 1. Overview
**MLTrain-Sched** is an ML-guided, workload-aware CPU scheduling framework designed specifically for machine learning training workloads, built upon Linux's **`sched_ext`** framework. It bridges classical OS scheduling principles with modern lightweight machine learning and deterministic fairness safeguards.

---

## 2. Two-Plane Architecture

```
                          USER SPACE (Control Plane)
 ┌────────────────────────────────────────────────────────────────────────┐
 │                                                                        │
 │  Workload Manager (PyTorch / Compute Stress / I/O Burst)               │
 │         │                                                              │
 │         ▼                                                              │
 │  Telemetry Collector (/proc/pid/stat, perf stat, eBPF tracepoints)     │
 │         │                                                              │
 │         ▼                                                              │
 │  Feature Extraction Engine (CPU/IO ratios, burst EMA, context switches)│
 │         │                                                              │
 │         ▼                                                              │
 │  Lightweight Random Forest Classifier (100 Decision Trees)             │
 │         │                                                              │
 │         ▼                                                              │
 │  Policy Controller:                                                    │
 │     - Confidence Check (if conf < 0.65 -> Rule-based fallback)         │
 │     - Continuous Aging (starvation prevention)                         │
 │     - Adaptive Quantum Assignment (4ms - 20ms)                         │
 │     - Multi-Level DSQ Mapping (HIGH, MED, LOW)                         │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │
                 BPF Map Updates (task_state_map & policy_cfg)
                                     │
                                     ▼
                     KERNEL SPACE (eBPF / sched_ext)
 ┌────────────────────────────────────────────────────────────────────────┐
 │  scx_mltrain (struct sched_ext_ops)                                    │
 │                                                                        │
 │   ├── select_cpu(): Preserves core cache affinity & selects idle CPUs   │
 │   ├── enqueue(): Dispatches task into assigned DSQ with dynamic slice  │
 │   ├── dispatch(): Drains queues in strict multi-level order:            │
 │   │               1. DSQ_HIGH  (ML Training compute loops / I/O yields)│
 │   │               2. DSQ_MED   (Mixed & general tasks)                 │
 │   │               3. DSQ_LOW   (Background CPU-bound contention)       │
 │   ├── running(): Records tick timestamp & increments context-switch ctr│
 │   └── stopping(): Updates accumulated runtime per task                 │
 └────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Dispatch Queues (DSQs)
Under Linux `sched_ext`, tasks are not scheduled onto CPUs directly from a single global runqueue; instead, they are routed through Dispatch Queues (DSQs):
* **`DSQ_HIGH` (ID 0):** High-priority queue reserved for active ML training workers during compute phases, interactive I/O tasks requiring immediate completion, and starving tasks promoted via aging.
* **`DSQ_MEDIUM` (ID 1):** Default queue for mixed workloads and normal priority processes.
* **`DSQ_LOW` (ID 2):** Background queue for CPU-intensive stress processes to prevent them from seizing cores from training threads.

---

## 4. Adaptive Time Slicing (Quantum)
Fixed round-robin quantums (e.g. CFS default 4ms) inflict heavy context-switching overhead on ML training loops due to cache flushes and synchronization latency. MLTrain-Sched dynamically adapts the time slice:
* **`CPU_BOUND` Tasks:** Assigned a long quantum ($18\,\text{ms} - 20\,\text{ms}$) to maximize CPU cache locality.
* **`IO_BOUND` Tasks:** Assigned a short quantum ($4\,\text{ms} - 5\,\text{ms}$) to yield CPU quickly and initiate asynchronous disk/network transfers.
* **`ML_TRAINING` Tasks:** Assigned an optimized compute quantum ($14\,\text{ms} - 16\,\text{ms}$).
* **`MIXED` Tasks:** Standard quantum ($10\,\text{ms}$).

---

## 5. Starvation Prevention & Continuous Aging
To prevent high-priority ML tasks from indefinitely starving background processes:
$$\text{effective\_score} = \text{ml\_score} + \alpha \times \frac{\text{waiting\_time}}{\text{starvation\_threshold}}$$
If a task waits longer than a hard threshold (e.g., 3,000 ms), the kernel scheduler automatically elevates it to `DSQ_HIGH`.
