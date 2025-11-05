#!/bin/bash

# RCCL Multi-Node Performance Test Script
# This script tests RCCL communication between two nodes using multiple high-speed interfaces

set -e

# Network interfaces to use (eth2-eth9, 400Gbps each)
export RCCL_SOCKET_IFNAME=eth2,eth3,eth4,eth5,eth6,eth7,eth8,eth9

# Enable debug information
export RCCL_DEBUG=INFO
export RCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV,NET

# Configure RCCL for optimal performance
export RCCL_NET_GDR_LEVEL=1  # GPU Direct RDMA
export RCCL_IB_DISABLE=1      # Disable InfiniBand (use Ethernet)
export RCCL_NET_SHARED_BUFFERS=1

# MPI settings for distributed execution
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

# Test parameters
TEST_BINARY="/tmp/rccl-tests/build/all_reduce_perf"
HOSTFILE="/tmp/hostfile"
NODES=2
GPUS_PER_NODE=8
TOTAL_GPUS=$((NODES * GPUS_PER_NODE))

echo "=== RCCL Multi-Node Performance Test ==="
echo "Nodes: $NODES"
echo "GPUs per node: $GPUS_PER_NODE"
echo "Total GPUs: $TOTAL_GPUS"
echo "Network interfaces: $RCCL_SOCKET_IFNAME"
echo "=================================="

# Function to run test with specific parameters
run_test() {
    local test_name=$1
    local additional_args=$2
    
    echo ""
    echo "Running $test_name..."
    echo "Command: mpirun -np $TOTAL_GPUS -hostfile $HOSTFILE --allow-run-as-root $TEST_BINARY $additional_args"
    
    mpirun -np $TOTAL_GPUS -hostfile $HOSTFILE \
           --allow-run-as-root \
           --mca btl_tcp_if_include eth2,eth3,eth4,eth5,eth6,eth7,eth8,eth9 \
           --mca oob_tcp_if_include eth2,eth3,eth4,eth5,eth6,eth7,eth8,eth9 \
           $TEST_BINARY $additional_args
}

# Test 1: Small message sizes (latency test)
run_test "AllReduce Latency Test (small messages)" "-b 8 -e 1K -f 2 -g 1"

# Test 2: Large message sizes (bandwidth test)  
run_test "AllReduce Bandwidth Test (large messages)" "-b 1M -e 1G -f 2 -g 1"

# Test 3: Mixed message sizes
run_test "AllReduce Mixed Sizes" "-b 1K -e 16M -f 2 -g 1"

echo ""
echo "=== RCCL Test Complete ==="