# MLTrain-Sched: Course Project Viva & Technical Defense Reference

This document provides direct, authoritative answers to questions frequently asked during systems and operating systems course defenses.

---

## Part 1: Core Fundamentals

### Q1: What is `sched_ext` and why is it revolutionary for Linux scheduling?
**Answer:**
`sched_ext` is an upstream Linux scheduling class (introduced in Linux 6.12) that enables custom CPU scheduling policies to be implemented as eBPF programs and dynamically attached using `struct sched_ext_ops`.
Before `sched_ext`, experimenting with CPU scheduling required patching the in-tree CFS/EEVDF code in C, re-compiling the Linux kernel, and rebooting. Any bug caused a kernel panic. With `sched_ext`:
1. Schedulers are loaded dynamically from user space like kernel modules.
2. The eBPF verifier mathematically guarantees memory safety.
3. If an error occurs, the kernel safely aborts and restores default CFS/EEVDF scheduling without a system crash.

---

### Q2: Why run ML inference in user space instead of directly inside eBPF?
**Answer:**
1. **BPF Verifier Limitations:** The eBPF verifier strictly limits program complexity (maximum 1 million verified instructions, 512 bytes of stack space, bounded loops, and no floating-point arithmetic). Complex ML models cannot pass verifier analysis.
2. **Scheduling Latency:** Kernel dispatch paths run in nanoseconds to single-digit microseconds. Evaluating a multi-tree Random Forest on every enqueue would saturate the CPU.
3. **Decoupled Architecture:** User space samples telemetry and updates compact BPF hash maps (`task_state_map`) at 100ms–1s intervals. The BPF kernel code performs $O(1)$ lookups and queue dispatches.

---

### Q3: What is a DSQ in `sched_ext`?
**Answer:**
A **DSQ (Dispatch Queue)** is a scheduling queue abstraction provided by `sched_ext` to organize runnable tasks. In MLTrain-Sched, we allocate three DSQs:
* `DSQ_HIGH` (0): High-priority ML training tasks, interactive I/O tasks, and aging promotions.
* `DSQ_MEDIUM` (1): Standard mixed tasks.
* `DSQ_LOW` (2): Heavy background CPU stress tasks.
The `dispatch` callback consumes tasks from `DSQ_HIGH` first, then `DSQ_MEDIUM`, and lastly `DSQ_LOW`.

---

## Part 2: Scheduling Policy, ML, and Fairness

### Q4: Why Random Forest over Deep Neural Networks?
**Answer:**
1. **Tabular Telemetry:** CPU burst, I/O rates, context switches, and runtime are tabular metrics where tree ensembles consistently outperform neural networks.
2. **Deterministic Latency:** Decision trees require only simple conditional comparisons with zero matrix multiplication overhead.
3. **Interpretability:** Random Forest outputs explicit feature importances and class probabilities, allowing us to explain why each task was placed into a specific DSQ.

---

### Q5: How is starvation prevented when ML prioritizes ML workloads?
**Answer:**
Through **Continuous Aging**:
$$\text{effective\_score} = \text{ml\_score} + \alpha \times \frac{\text{waiting\_time}}{\text{starvation\_limit}}$$
As a lower-priority task remains in the ready queue, its waiting time monotonically increases its score. If wait time exceeds the starvation limit, it is automatically promoted into `DSQ_HIGH`.
In our ablation study (A2: No Aging), maximum waiting time spiked from 24,933 to 31,120 ticks, experimentally validating that aging is necessary for system liveness.

---

### Q6: What is Jain's Fairness Index and how did you use it?
**Answer:**
Jain's Fairness Index evaluates how equitably resources are distributed among $n$ processes:
$$J = \frac{\left(\sum_{i=1}^n x_i\right)^2}{n \sum_{i=1}^n x_i^2}$$
* $J = 1.0$: Absolute equality.
* $J = \frac{1}{n}$: Worst-case inequality (one process monopolizes everything).
In MLTrain-Sched, we calculate $J$ on the normalized waiting time balance ($1 - w_i / w_{\max}$) and CPU service received. MLTrain-Sched achieves $J \approx 0.742$, maintaining fair CPU access alongside classical schedulers ($0.748$).

---

### Q7: What happens when the ML model makes an incorrect prediction?
**Answer:**
We implement a **Two-Tier Safety Net**:
1. **Confidence-Aware Fallback:** If the classifier's maximum probability $\max(P) < \tau$ (e.g. 0.65), the scheduler falls back to a deterministic rule-based assignment based on CPU vs I/O burst ratios.
2. **Adaptive Preemption Bounds:** Even if a background task is misclassified as ML training, its time slice is capped at $20\,\text{ms}$, and other tasks continue to receive CPU time via Round Robin queue cycling and aging.
