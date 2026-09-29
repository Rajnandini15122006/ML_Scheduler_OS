# Makefile for scx_mltrain (sched_ext BPF scheduler)

CLANG ?= clang
BPFTOOL ?= bpftool
CFLAGS ?= -g -O2 -Wall

all: scx_mltrain

# Generate vmlinux.h if missing
vmlinux.h:
	$(BPFTOOL) btf dump file /sys/kernel/btf/vmlinux format c > vmlinux.h 2>/dev/null || touch vmlinux.h

# Compile BPF C code
scx_mltrain.bpf.o: scx_mltrain.bpf.c scx_mltrain.h vmlinux.h
	$(CLANG) -g -O2 -target bpf -D__TARGET_ARCH_x86 -I. -c scx_mltrain.bpf.c -o scx_mltrain.bpf.o

# Generate BPF skeleton
scx_mltrain.skel.h: scx_mltrain.bpf.o
	$(BPFTOOL) gen skeleton scx_mltrain.bpf.o > scx_mltrain.skel.h

# Compile User-Space binary
scx_mltrain: scx_mltrain.c scx_mltrain.skel.h scx_mltrain.h
	$(CC) $(CFLAGS) -I. scx_mltrain.c -lbpf -lelf -lz -o scx_mltrain

clean:
	rm -f scx_mltrain.bpf.o scx_mltrain.skel.h scx_mltrain vmlinux.h

.PHONY: all clean
