#!/bin/bash

# Simple RCCL Multi-Node Test Script
echo "=== Starting RCCL Multi-Node Test ==="
echo "Testing communication between:"
echo "  Node 1: 192.168.50.2 (8 GPUs)"
echo "  Node 2: 192.168.50.3 (8 GPUs)"
echo "Using high-speed interfaces eth2-eth9 (400Gbps each)"
echo "=================================="

# Set environment variables
export RCCL_SOCKET_IFNAME=eth2,eth3,eth4,eth5,eth6,eth7,eth8,eth9
export RCCL_DEBUG=WARN
export RCCL_NET_GDR_LEVEL=1
export RCCL_IB_DISABLE=1
export OMPI_ALLOW_RUN_AS_ROOT=1

# Test 1: 2 GPUs (1 per node)
echo ""
echo "Test 1: AllReduce with 2 GPUs (1 per node)"
echo "Command: mpirun -np 2 -H 192.168.50.2:1,192.168.50.3:1 --allow-run-as-root /tmp/rccl-tests/build/all_reduce_perf -b 1M -e 64M -f 2 -g 1"

mpirun -np 2 \
       -H 192.168.50.2:1,192.168.50.3:1 \
       --allow-run-as-root \
       --mca btl_tcp_if_include eth2,eth3,eth4,eth5,eth6,eth7,eth8,eth9 \
       --mca oob_tcp_if_include eth2,eth3,eth4,eth5,eth6,eth7,eth8,eth9 \
       /tmp/rccl-tests/build/all_reduce_perf -b 1M -e 64M -f 2 -g 1

echo ""
echo "=== Test Complete ==="