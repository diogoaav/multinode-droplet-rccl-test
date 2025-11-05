#!/bin/bash

# Comprehensive RCCL Multi-Node Performance Test
echo "=== RCCL Multi-Node Performance Benchmark ==="
echo "Testing 400Gbps interconnect performance"
echo "Network: eth2-eth9 (8x 400Gbps interfaces)"
echo "Node 1: 192.168.50.2 (8x MI325X GPUs)"
echo "Node 2: 192.168.50.3 (8x MI325X GPUs)"
echo "=============================================="

# Environment setup
export RCCL_SOCKET_IFNAME=eth2,eth3,eth4,eth5,eth6,eth7,eth8,eth9
export RCCL_DEBUG=WARN
export RCCL_NET_GDR_LEVEL=1
export RCCL_IB_DISABLE=1
export OMPI_ALLOW_RUN_AS_ROOT=1

# Function to run test
run_test() {
    local test_name="$1"
    local np="$2"
    local node_spec="$3"
    local args="$4"
    
    echo ""
    echo "=== $test_name ==="
    echo "Processes: $np"
    echo "Node specification: $node_spec"
    echo "Test parameters: $args"
    echo ""
    
    mpirun -np $np \
           -H $node_spec \
           --allow-run-as-root \
           --mca btl_tcp_if_include eth2,eth3,eth4,eth5,eth6,eth7,eth8,eth9 \
           --mca oob_tcp_if_include eth2,eth3,eth4,eth5,eth6,eth7,eth8,eth9 \
           /tmp/rccl-tests/build/all_reduce_perf $args
    
    echo "Test completed: $test_name"
    echo ""
}

# Test 1: 2 GPUs (1 per node) - Latency test
run_test "AllReduce Latency (2 GPUs)" \
         "2" \
         "192.168.50.2:1,192.168.50.3:1" \
         "-b 8 -e 1K -f 2 -g 1"

# Test 2: 4 GPUs (2 per node) - Medium bandwidth
run_test "AllReduce Medium Scale (4 GPUs)" \
         "4" \
         "192.168.50.2:2,192.168.50.3:2" \
         "-b 1M -e 64M -f 2 -g 1"

# Test 3: 8 GPUs (4 per node) - High bandwidth
run_test "AllReduce High Scale (8 GPUs)" \
         "8" \
         "192.168.50.2:4,192.168.50.3:4" \
         "-b 4M -e 256M -f 2 -g 1"

# Test 4: All 16 GPUs - Maximum bandwidth test
run_test "AllReduce Maximum Scale (16 GPUs)" \
         "16" \
         "192.168.50.2:8,192.168.50.3:8" \
         "-b 16M -e 512M -f 2 -g 1"

echo "=== All RCCL Tests Complete ==="
echo "Performance summary shows utilization of 400Gbps interconnect"