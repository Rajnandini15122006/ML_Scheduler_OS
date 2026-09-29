/*
 * scx_mltrain.c - User-space loader and control plane for scx_mltrain
 */

#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>
#include "scx_mltrain.h"
#include "scx_mltrain.skel.h"

static volatile bool exiting = false;

static void sig_handler(int sig)
{
    exiting = true;
}

int main(int argc, char **argv)
{
    struct scx_mltrain_bpf *skel;
    int err;

    signal(SIGINT, sig_handler);
    signal(SIGTERM, sig_handler);

    printf("===================================================\n");
    printf(" MLTRAIN-SCHED: User-Space Loader & Controller\n");
    printf(" Built on Linux sched_ext\n");
    printf("===================================================\n");

    /* Open and load BPF skeleton */
    skel = scx_mltrain_bpf__open();
    if (!skel) {
        fprintf(stderr, "Failed to open BPF skeleton\n");
        return 1;
    }

    err = scx_mltrain_bpf__load(skel);
    if (err) {
        fprintf(stderr, "Failed to load and verify BPF skeleton: %d\n", err);
        goto cleanup;
    }

    /* Initialize default policy config */
    struct policy_config cfg = {
        .aging_factor = 25,
        .max_wait_ms = 100,
        .confidence_thresh = 70,
        .feedback_enabled = 1,
        .min_quantum_us = MIN_QUANTUM_US,
        .max_quantum_us = MAX_QUANTUM_US,
    };
    unsigned int zero = 0;
    bpf_map_update_elem(bpf_map__fd(skel->maps.policy_cfg), &zero, &cfg, BPF_ANY);

    /* Attach sched_ext scheduler */
    err = scx_mltrain_bpf__attach(skel);
    if (err) {
        fprintf(stderr, "Failed to attach sched_ext ops: %d\n", err);
        goto cleanup;
    }

    printf("Successfully registered scx_mltrain with kernel!\n");
    printf("Dispatch Queues initialized: DSQ_HIGH(0), DSQ_MED(1), DSQ_LOW(2)\n");
    printf("Listening for workload events... Press Ctrl+C to exit.\n");

    while (!exiting) {
        sleep(1);
    }

    printf("\nUnregistering scx_mltrain and restoring default Linux scheduler...\n");

cleanup:
    scx_mltrain_bpf__destroy(skel);
    return err ? 1 : 0;
}
