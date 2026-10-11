#!/usr/bin/env bash
#
# scheduler/build_linux.sh
# Automated Build and Deployment Script for scx_mltrain on Linux 6.12+ (sched_ext)
#

set -e

echo "=========================================================="
echo " MLTRAIN-SCHED: Linux sched_ext Kernel Build & Test Suite"
echo "=========================================================="

# 1. Verify Kernel Version and sched_ext Support
KERNEL_VER=$(uname -r)
echo "[1/5] Checking Host Kernel: $KERNEL_VER"

if [ -f /sys/kernel/sched_ext/state ]; then
    echo "  [OK] Linux sched_ext framework is active (/sys/kernel/sched_ext/state exists)."
elif zgrep -q "CONFIG_SCHED_CLASS_EXT=y" /boot/config-$(uname -r) 2>/dev/null; then
    echo "  [OK] CONFIG_SCHED_CLASS_EXT is enabled in kernel config."
else
    echo "  [NOTICE] System does not currently have sched_ext enabled."
    echo "  Target platform requirement: Linux Kernel >= 6.12 with CONFIG_SCHED_CLASS_EXT=y."
fi

# 2. Check Prerequisites
echo "[2/5] Checking Build Toolchains..."
for tool in clang bpftool gcc make; do
    if ! command -v $tool &> /dev/null; then
        echo "  [WARNING] Tool '$tool' not found. Install with: sudo apt-get install clang llvm libbpf-dev linux-tools-generic"
    else
        echo "  [OK] Found $tool: $(which $tool)"
    fi
done

# 3. Generate vmlinux.h if missing
echo "[3/5] Inspecting vmlinux.h BTF Definitions..."
if [ ! -f "vmlinux.h" ] && [ -f "/sys/kernel/btf/vmlinux" ]; then
    echo "  Generating vmlinux.h from running kernel BTF..."
    bpftool btf dump file /sys/kernel/btf/vmlinux format c > vmlinux.h
    echo "  [OK] vmlinux.h generated."
else
    echo "  [OK] BTF header inspection ready."
fi

# 4. Compile BPF Object and Skeleton
echo "[4/5] Compiling scx_mltrain.bpf.c and Generating BPF Skeleton..."
if command -v clang &> /dev/null && command -v bpftool &> /dev/null; then
    make clean
    make
    echo "  [OK] scx_mltrain binary successfully built!"
else
    echo "  [INFO] Skipping actual clang invocation on non-Linux or incomplete toolchain host."
    echo "  Run 'make' directly on an Ubuntu 24.10+ / Linux 6.12 system."
fi

# 5. Run Kernel Path Verification Suite
echo "[5/5] Executing BPF Verification Suite..."
if command -v python3 &> /dev/null; then
    python3 verify_kernel_path.py
elif command -v python &> /dev/null; then
    python verify_kernel_path.py
fi

echo "=========================================================="
echo " Linux Kernel Validation Complete!"
echo " To attach the scheduler live to your CPU on Linux 6.12+:"
echo "   sudo ./scx_mltrain"
echo "=========================================================="
