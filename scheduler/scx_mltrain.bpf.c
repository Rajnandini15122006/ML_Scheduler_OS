/*
 * scx_mltrain.bpf.c - An ML-Guided CPU Scheduler for ML Training Workloads
 * Built on Linux sched_ext Framework
 */

#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include "scx_mltrain.h"

char _license[] SEC("license") = "GPL";

/*
 * BPF Map: Task State Map (populated by user-space telemetry & updated by BPF)
 */
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 10240);
    __type(key, u32); /* PID / TID */
    __type(value, struct task_state);
} task_state_map SEC(".maps");

/*
 * BPF Map: Global Policy Configuration
 */
struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __type(key, u32);
    __type(value, struct policy_config);
} policy_cfg SEC(".maps");

/* Helper to get current policy configuration */
static inline struct policy_config *get_config(void)
{
    u32 zero = 0;
    return bpf_map_lookup_elem(&policy_cfg, &zero);
}

/*
 * select_cpu: Choose an optimal CPU for a runnable task
 */
s32 BPF_STRUCT_OPS(mltrain_select_cpu, struct task_struct *p, s32 prev_cpu, u64 wake_flags)
{
    /* Prioritize idle CPU or keep prev_cpu for cache locality */
    return prev_cpu;
}

/*
 * enqueue: Place runnable task into appropriate DSQ with aging protection
 */
void BPF_STRUCT_OPS(mltrain_enqueue, struct task_struct *p, u64 enq_flags)
{
    u32 pid = p->pid;
    u64 now = bpf_ktime_get_ns();
    u64 dsq_id = DSQ_MEDIUM;
    u64 slice_us = DEFAULT_QUANTUM_US;

    struct task_state *ts = bpf_map_lookup_elem(&task_state_map, &pid);
    struct policy_config *cfg = get_config();

    if (ts) {
        ts->enqueue_count++;
        u64 wait_ns = 0;
        if (ts->last_start_ns > 0 && now > ts->last_start_ns) {
            wait_ns = now - ts->last_start_ns;
            ts->total_wait_ns += wait_ns;
        }

        /* Aging calculation: Starvation protection */
        u64 max_wait_ns = (cfg ? (u64)cfg->max_wait_ms : 100ULL) * 1000000ULL;
        if (wait_ns > max_wait_ns) {
            /* Promote starved task to HIGH priority DSQ */
            dsq_id = DSQ_HIGH;
            slice_us = ts->target_quantum_us ? ts->target_quantum_us : DEFAULT_QUANTUM_US;
        } else {
            /* Map predicted workload class to DSQ & adaptive quantum */
            switch (ts->workload_class) {
                case CLASS_ML_TRAINING:
                    dsq_id = DSQ_HIGH;
                    slice_us = 15000; /* 15 ms compute quantum */
                    break;
                case CLASS_CPU_BOUND:
                    dsq_id = (ts->ml_priority >= 70) ? DSQ_HIGH : DSQ_MEDIUM;
                    slice_us = 20000; /* 20 ms throughput quantum */
                    break;
                case CLASS_IO_BOUND:
                    dsq_id = DSQ_HIGH; /* Rapid response for I/O completion */
                    slice_us = 4000;  /* 4 ms interactive quantum */
                    break;
                case CLASS_MIXED:
                default:
                    dsq_id = DSQ_MEDIUM;
                    slice_us = DEFAULT_QUANTUM_US;
                    break;
            }
        }
        ts->assigned_dsq = (u32)dsq_id;
    }

    /* Enqueue to target DSQ with slice */
    scx_bpf_dispatch(p, dsq_id, slice_us * 1000ULL, enq_flags);
}

/*
 * dispatch: Move tasks from DSQs to CPU
 */
void BPF_STRUCT_OPS(mltrain_dispatch, s32 cpu, struct task_struct *prev)
{
    /* Strict priority dispatch: HIGH -> MEDIUM -> LOW */
    if (scx_bpf_consume(DSQ_HIGH))
        return;
    if (scx_bpf_consume(DSQ_MEDIUM))
        return;
    scx_bpf_consume(DSQ_LOW);
}

/*
 * running: Track when task starts running on CPU
 */
void BPF_STRUCT_OPS(mltrain_running, struct task_struct *p)
{
    u32 pid = p->pid;
    u64 now = bpf_ktime_get_ns();
    struct task_state *ts = bpf_map_lookup_elem(&task_state_map, &pid);
    if (ts) {
        ts->last_start_ns = now;
        ts->ctx_switch_count++;
    }
}

/*
 * stopping: Track when task yields or is preempted
 */
void BPF_STRUCT_OPS(mltrain_stopping, struct task_struct *p, bool runnable)
{
    u32 pid = p->pid;
    u64 now = bpf_ktime_get_ns();
    struct task_state *ts = bpf_map_lookup_elem(&task_state_map, &pid);
    if (ts && ts->last_start_ns > 0) {
        u64 delta = now - ts->last_start_ns;
        ts->total_runtime_ns += delta;
    }
}

/*
 * init: Create DSQs
 */
s32 BPF_STRUCT_OPS_ENTRY(mltrain_init)
{
    s32 ret;

    ret = scx_bpf_create_dsq(DSQ_HIGH, -1);
    if (ret)
        return ret;

    ret = scx_bpf_create_dsq(DSQ_MEDIUM, -1);
    if (ret)
        return ret;

    ret = scx_bpf_create_dsq(DSQ_LOW, -1);
    if (ret)
        return ret;

    return 0;
}

/*
 * exit: Scheduler unload cleanup
 */
void BPF_STRUCT_OPS(mltrain_exit, struct scx_exit_info *info)
{
    /* Clean up maps or dump telemetry */
}

/*
 * sched_ext operations table
 */
SEC(".struct_ops.link")
struct sched_ext_ops mltrain_ops = {
    .select_cpu         = mltrain_select_cpu,
    .enqueue            = mltrain_enqueue,
    .dispatch           = mltrain_dispatch,
    .running            = mltrain_running,
    .stopping           = mltrain_stopping,
    .init               = mltrain_init,
    .exit               = mltrain_exit,
    .name               = "scx_mltrain",
};
