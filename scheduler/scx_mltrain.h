#ifndef __SCX_MLTRAIN_H
#define __SCX_MLTRAIN_H

/* Dispatch Queue (DSQ) Identifiers */
#define DSQ_HIGH    0
#define DSQ_MEDIUM  1
#define DSQ_LOW     2

/* Default Quantum Bounds (in microseconds) */
#define MIN_QUANTUM_US      1000    /* 1 ms */
#define DEFAULT_QUANTUM_US  5000    /* 5 ms */
#define MAX_QUANTUM_US      20000   /* 20 ms */

/* Workload Classifications */
typedef enum {
    CLASS_CPU_BOUND    = 0,
    CLASS_IO_BOUND     = 1,
    CLASS_ML_TRAINING  = 2,
    CLASS_MIXED        = 3,
    CLASS_UNKNOWN      = 4
} workload_class_t;

/* Logical Task State tracked by BPF and User-Space */
struct task_state {
    unsigned long long last_start_ns;
    unsigned long long total_runtime_ns;
    unsigned long long total_wait_ns;
    unsigned long long enqueue_count;
    unsigned long long wakeup_count;
    unsigned long long ctx_switch_count;

    unsigned int workload_class;
    unsigned int ml_priority;        /* 0 - 100 */
    unsigned int target_quantum_us;  /* Dynamic time slice */
    unsigned int assigned_dsq;       /* 0: HIGH, 1: MED, 2: LOW */
    unsigned int flags;
};

/* Global Policy Configuration synced via BPF Map */
struct policy_config {
    unsigned int aging_factor;       /* Weight for wait time aging */
    unsigned int max_wait_ms;        /* Starvation threshold */
    unsigned int confidence_thresh;  /* Scaled 0-100 */
    unsigned int feedback_enabled;   /* 0 or 1 */
    unsigned int min_quantum_us;
    unsigned int max_quantum_us;
};

#endif /* __SCX_MLTRAIN_H */
