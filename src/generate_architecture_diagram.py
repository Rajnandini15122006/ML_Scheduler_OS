"""
src/generate_architecture_diagram.py
Generates a publication-grade system architecture diagram for MLTrain-Sched.
Saved as results/system_architecture.png.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def generate_diagram():
    fig, ax = plt.subplots(figsize=(15, 10.5), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    # Header
    ax.text(50, 97.5, "MLTRAIN-SCHED: SYSTEM ARCHITECTURE & DATA FLOW", 
            fontsize=17, fontweight="bold", ha="center", color="#0F172A", family="sans-serif")
    ax.text(50, 94.8, "Decoupled Two-Plane Architecture: User-Space ML Telemetry + Linux sched_ext Kernel BPF", 
            fontsize=10.5, ha="center", color="#475569", family="sans-serif")

    # -------------------------------------------------------------
    # 1. USER SPACE PLANE (Top Box)
    # -------------------------------------------------------------
    user_plane = patches.FancyBboxPatch(
        (5, 52), 90, 40,
        boxstyle="round,pad=1.2,rounding_size=2",
        linewidth=1.5, edgecolor="#93C5FD", facecolor="#F8FAFC"
    )
    ax.add_patch(user_plane)
    ax.text(8, 89, "USER-SPACE CONTROL PLANE (Python Runtime & ML Daemon)", 
            fontsize=11.5, fontweight="bold", color="#1E40AF")

    # A. Workload Suite
    box_workload = patches.FancyBboxPatch((8, 69), 24, 16.5, boxstyle="round,pad=0.8,rounding_size=1",
                                          edgecolor="#CBD5E1", facecolor="#FFFFFF", linewidth=1.2)
    ax.add_patch(box_workload)
    ax.text(20, 82, "1. WORKLOAD SUITE", fontsize=9.5, fontweight="bold", ha="center", color="#0F172A")
    ax.text(20, 75.5, "• PyTorch ML Training Loop\n• Synthetic Contention\n• Data Loaders & I/O Tasks", 
            fontsize=8.5, ha="center", color="#475569", linespacing=1.4)

    # B. Telemetry & Features
    box_telemetry = patches.FancyBboxPatch((38, 69), 25, 16.5, boxstyle="round,pad=0.8,rounding_size=1",
                                           edgecolor="#CBD5E1", facecolor="#FFFFFF", linewidth=1.2)
    ax.add_patch(box_telemetry)
    ax.text(50.5, 82, "2. TELEMETRY & FEATURES", fontsize=9.5, fontweight="bold", ha="center", color="#0F172A")
    ax.text(50.5, 75.5, "• /proc/[pid]/stat, perf stat\n• CPU/IO Burst Ratios\n• Context Switches & Waiting", 
            fontsize=8.5, ha="center", color="#475569", linespacing=1.4)

    # C. Random Forest Classifier
    box_ml = patches.FancyBboxPatch((68, 69), 24, 16.5, boxstyle="round,pad=0.8,rounding_size=1",
                                    edgecolor="#3B82F6", facecolor="#EFF6FF", linewidth=1.5)
    ax.add_patch(box_ml)
    ax.text(80, 82, "3. RANDOM FOREST ML", fontsize=9.5, fontweight="bold", ha="center", color="#1D4ED8")
    ax.text(80, 75.5, "• 100 Tabular Trees\n• Multi-Class Probabilities\n• Inference Confidence", 
            fontsize=8.5, ha="center", color="#1E3A8A", linespacing=1.4)

    # D. Policy Engine
    box_policy = patches.FancyBboxPatch((18, 54), 65, 10.5, boxstyle="round,pad=0.8,rounding_size=1",
                                        edgecolor="#10B981", facecolor="#F0FDF4", linewidth=1.2)
    ax.add_patch(box_policy)
    ax.text(50.5, 61.5, "4. ADAPTIVE SCHEDULING POLICY ENGINE", fontsize=9.5, fontweight="bold", ha="center", color="#065F46")
    ax.text(50.5, 56.5, "Assigns: DSQ Level (HIGH/MED/LOW) | Adaptive Quantum (4-20ms) | Starvation Aging Score", 
            fontsize=8.5, ha="center", color="#047857")

    # Connectors User Space
    ax.annotate("", xy=(38, 77.2), xytext=(32, 77.2), arrowprops=dict(arrowstyle="->", color="#3B82F6", lw=1.5))
    ax.annotate("", xy=(68, 77.2), xytext=(63, 77.2), arrowprops=dict(arrowstyle="->", color="#3B82F6", lw=1.5))
    ax.annotate("", xy=(50.5, 64.5), xytext=(50.5, 69), arrowprops=dict(arrowstyle="->", color="#10B981", lw=1.5))

    # -------------------------------------------------------------
    # 2. CONTROL INTERFACE / BPF MAPS (Middle Bridge)
    # -------------------------------------------------------------
    bridge = patches.FancyBboxPatch(
        (15, 41), 71, 7.5,
        boxstyle="round,pad=0.8,rounding_size=1.5",
        linewidth=1.5, edgecolor="#F59E0B", facecolor="#FFFBEB"
    )
    ax.add_patch(bridge)
    ax.text(50.5, 45.8, "TWO-WAY CONTROL INTERFACE (BPF SHARED MEMORY MAPS)", 
            fontsize=9.5, fontweight="bold", ha="center", color="#B45309")
    ax.text(50.5, 42.5, "task_state_map (Per-Task Priority & Slices)  |  policy_cfg (Global Bounds & Aging Config)", 
            fontsize=8.5, ha="center", color="#92400E")

    ax.annotate("", xy=(50.5, 48.5), xytext=(50.5, 54), arrowprops=dict(arrowstyle="->", color="#B45309", lw=2))
    ax.annotate("", xy=(50.5, 36.5), xytext=(50.5, 41), arrowprops=dict(arrowstyle="->", color="#B45309", lw=2))

    # -------------------------------------------------------------
    # 3. KERNEL SPACE PLANE (Bottom Box)
    # -------------------------------------------------------------
    kernel_plane = patches.FancyBboxPatch(
        (5, 4), 90, 31,
        boxstyle="round,pad=1.2,rounding_size=2",
        linewidth=1.5, edgecolor="#CBD5E1", facecolor="#F8FAFC"
    )
    ax.add_patch(kernel_plane)
    ax.text(8, 31.8, "KERNEL SPACE (Linux sched_ext Framework & scx_mltrain eBPF Scheduler)", 
            fontsize=11.5, fontweight="bold", color="#334155")

    # BPF Callbacks Box
    box_ops = patches.FancyBboxPatch((8, 8), 24, 20, boxstyle="round,pad=0.8,rounding_size=1",
                                     edgecolor="#94A3B8", facecolor="#FFFFFF", linewidth=1.2)
    ax.add_patch(box_ops)
    ax.text(20, 24.5, "struct sched_ext_ops", fontsize=9.5, fontweight="bold", ha="center", color="#0F172A")
    ax.text(20, 16.5, "• select_cpu(): Locality\n• enqueue(): DSQ routing\n• dispatch(): Drain order\n• running() / stopping()\n• init() / exit()", 
            fontsize=8.5, ha="center", color="#475569", linespacing=1.35)

    # Multi-Queue DSQs Box
    box_dsq = patches.FancyBboxPatch((37, 7.5), 33, 21, boxstyle="round,pad=0.8,rounding_size=1",
                                     edgecolor="#6366F1", facecolor="#EEF2FF", linewidth=1.5)
    ax.add_patch(box_dsq)
    ax.text(53.5, 25.5, "MULTI-LEVEL DISPATCH QUEUES (DSQ)", fontsize=9.5, fontweight="bold", ha="center", color="#4338CA")
    
    # Sub-DSQs
    dsq1 = patches.Rectangle((39, 19.5), 29, 4.2, facecolor="#DCFCE7", edgecolor="#86EFAC")
    ax.add_patch(dsq1)
    ax.text(53.5, 21.6, "DSQ_HIGH: ML Training Loops & I/O Fast-Path", fontsize=8, fontweight="bold", ha="center", color="#166534")

    dsq2 = patches.Rectangle((39, 14.3), 29, 4.2, facecolor="#FEF9C3", edgecolor="#FDE047")
    ax.add_patch(dsq2)
    ax.text(53.5, 16.4, "DSQ_MEDIUM: Mixed Tasks & Standard Priorities", fontsize=8, fontweight="bold", ha="center", color="#854D0E")

    dsq3 = patches.Rectangle((39, 9.1), 29, 4.2, facecolor="#FEE2E2", edgecolor="#FCA5A5")
    ax.add_patch(dsq3)
    ax.text(53.5, 11.2, "DSQ_LOW: Background CPU-Bound Contention", fontsize=8, fontweight="bold", ha="center", color="#991B1B")

    # CPU Cores Box
    box_cpus = patches.FancyBboxPatch((75, 8), 17, 20, boxstyle="round,pad=0.8,rounding_size=1",
                                      edgecolor="#0F172A", facecolor="#FFFFFF", linewidth=1.5)
    ax.add_patch(box_cpus)
    ax.text(83.5, 24.5, "HARDWARE CPUS", fontsize=9.5, fontweight="bold", ha="center", color="#0F172A")
    ax.text(83.5, 16.5, "Core 0   Core 1\nCore 2   Core 3\n\nDirect Task\nExecution Slices", 
            fontsize=8.5, ha="center", color="#334155", linespacing=1.35)

    # Connectors Kernel Space
    ax.annotate("", xy=(37, 18), xytext=(32, 18), arrowprops=dict(arrowstyle="->", color="#6366F1", lw=1.5))
    ax.annotate("", xy=(75, 18), xytext=(70, 18), arrowprops=dict(arrowstyle="->", color="#0F172A", lw=2))

    # Feedback Loop (Right side clean curve)
    ax.annotate("", xy=(93, 72), xytext=(93, 20),
                arrowprops=dict(arrowstyle="->", color="#059669", lw=2, connectionstyle="arc3,rad=-0.25"))
    ax.text(97.5, 46, "Continuous Telemetry Feedback (Throughput & Wait Time)", 
            fontsize=8, color="#059669", fontweight="bold", rotation=270, ha="center")

    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", "system_architecture.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Architecture diagram successfully generated at: {out_path}")

if __name__ == "__main__":
    generate_diagram()
