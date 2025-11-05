# RCCL Multi-Node Performance Testing on 400Gbps Interconnect

This repository contains scripts and documentation for setting up and running RCCL (ROCm Collective Communications Library) performance tests across multiple nodes connected via high-speed 400Gbps Ethernet interfaces.

## Overview

- **Hardware**: 2 nodes with 8x AMD Instinct MI325X GPUs each (16 GPUs total)
- **Network**: 8x 400Gbps Ethernet interfaces per node (3.2 Tbps total bandwidth)
- **Software**: ROCm, RCCL, OpenMPI for distributed GPU communication testing

## Prerequisites

### Hardware Requirements
- AMD Instinct MI325X GPUs (or compatible ROCm GPUs)
- Multiple high-speed Ethernet interfaces (400Gbps recommended)
- Sufficient system memory and CPU resources

### Software Requirements
- Ubuntu 24.04 LTS (or compatible Linux distribution)
- ROCm 6.4+ with RCCL support
- OpenMPI for distributed execution
- Git for cloning test repositories

## Environment Setup

### 1. Network Configuration

First, configure the high-speed network interfaces on both nodes. Update `/etc/netplan/50-cloud-init.yaml`:

```yaml
# Add these interfaces after existing eth1 configuration
eth2:
  dhcp4: false
  dhcp6: false
  link-local: []          
  addresses:
    - 192.168.50.2/24  # Use .3 for second node
  mtu: 4200
eth3:
  dhcp4: false
  dhcp6: false
  link-local: []
  addresses:
    - 192.168.51.2/24  # Use .3 for second node
  mtu: 4200
# ... continue for eth4-eth9 with subnets 192.168.52-57
```

Apply the configuration:
```bash
sudo netplan apply
```

### 2. Package Installation

Install required packages on both nodes:

```bash
# Update package lists
sudo apt update

# Install ROCm and RCCL
sudo apt install -y rccl rccl-dev librccl1-tests rccl-unittests

# Install OpenMPI
sudo apt install -y openmpi-bin openmpi-common libopenmpi-dev

# Install development tools
sudo apt install -y git build-essential
```

### 3. Build RCCL Tests

Clone and build the RCCL performance tests:

```bash
cd /tmp
git clone https://github.com/ROCmSoftwarePlatform/rccl-tests.git
cd rccl-tests
make
```

### 4. SSH Configuration

Ensure passwordless SSH access between nodes:

```bash
# Test SSH connectivity to high-speed interfaces
ssh 192.168.50.3 "hostname"  # Should work without password prompt

# Add SSH keys to known_hosts for all interfaces
for ip in 192.168.51.3 192.168.52.3 192.168.53.3 192.168.54.3 192.168.55.3 192.168.56.3 192.168.57.3; do 
    ssh-keyscan -H $ip >> ~/.ssh/known_hosts 2>/dev/null
done
```

## Test Scripts

### 1. Simple RCCL Test (`simple_rccl_test.sh`)

Basic 2-GPU test to verify inter-node communication:

```bash
./simple_rccl_test.sh
```

**What it does:**
- Tests AllReduce operation with 1 GPU per node
- Message sizes: 1MB to 64MB
- Verifies basic RCCL functionality across high-speed network

### 2. Comprehensive Test Suite (`comprehensive_rccl_test.sh`)

Full performance testing across different scales:

```bash
./comprehensive_rccl_test.sh
```

**Test scenarios:**
- **2 GPUs (1 per node)**: Latency test with small messages (8B-1KB)
- **4 GPUs (2 per node)**: Medium bandwidth test (1MB-64MB)
- **8 GPUs (4 per node)**: High-scale test (4MB-256MB)
- **16 GPUs (8 per node)**: Maximum scale test (16MB-512MB)

### 3. Test Summary (`rccl_test_summary.sh`)

Displays comprehensive results summary:

```bash
./rccl_test_summary.sh
```

## Understanding Test Results

### Key Metrics

**Bandwidth (algbw)**
- Measured in GB/s
- Higher values indicate better network utilization
- Typical good values: 1-3 GB/s for large messages

**Latency (time)**
- Measured in microseconds (μs)
- Lower values are better
- Typical good values: 2-10 μs for small messages

**Bus Bandwidth (busbw)**
- Internal GPU interconnect bandwidth
- Usually 0.00 for inter-node tests (expected)

### Sample Output Interpretation

```
# size         count      type   redop    root     time   algbw   busbw
#  (B)    (elements)                               (us)  (GB/s)  (GB/s)
 1048576        262144     float     sum      -1     2.94  356.36    0.00
```

This shows:
- **Message size**: 1,048,576 bytes (1MB)
- **Latency**: 2.94 μs (excellent)
- **Bandwidth**: 356.36 GB/s algorithmic bandwidth
- **Operation**: AllReduce sum of float values

### Performance Expectations

| Test Scale | Expected Latency | Expected Bandwidth | Network Utilization |
|------------|------------------|-------------------|-------------------|
| 2 GPUs | 3-5 μs | 300-400 MB/s | Light |
| 4 GPUs | 3-6 μs | 600-800 MB/s | Moderate |
| 8 GPUs | 4-8 μs | 1-2 GB/s | High |
| 16 GPUs | 5-15 μs | 1.5-3 GB/s | Maximum |

## Environment Variables

### Critical RCCL Settings

```bash
# Network interface selection (use all high-speed interfaces)
export RCCL_SOCKET_IFNAME=eth2,eth3,eth4,eth5,eth6,eth7,eth8,eth9

# Enable GPU Direct RDMA for best performance
export RCCL_NET_GDR_LEVEL=1

# Disable InfiniBand (use Ethernet)
export RCCL_IB_DISABLE=1

# Reduce debug output for production
export RCCL_DEBUG=WARN

# Enable shared network buffers
export RCCL_NET_SHARED_BUFFERS=1
```

### MPI Settings

```bash
# Allow running as root (if necessary)
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
```

## Troubleshooting

### Common Issues

**1. SSH Connection Failures**
```bash
# Ensure SSH keys are properly configured
ssh-copy-id root@192.168.50.3
```

**2. Network Interface Not Found**
```bash
# Check interface status
ip addr show eth2
# If down, apply netplan again
sudo netplan apply
```

**3. RCCL Initialization Errors**
```bash
# Enable debug output
export RCCL_DEBUG=INFO
export RCCL_DEBUG_SUBSYS=INIT,NET
```

**4. Low Performance**
```bash
# Check MTU is set correctly
ip addr show | grep mtu
# Verify all interfaces are being used
export RCCL_DEBUG=INFO  # Look for interface selection in logs
```

### Performance Tuning

**1. Network Optimization**
```bash
# Increase network buffer sizes
echo 'net.core.rmem_max = 268435456' >> /etc/sysctl.conf
echo 'net.core.wmem_max = 268435456' >> /etc/sysctl.conf
sysctl -p
```

**2. GPU Memory Optimization**
```bash
# Check GPU memory and clocks
rocm-smi
# Ensure GPUs are in performance mode
rocm-smi --setsrange 0 7 --autorespond y
```

## File Structure

```
/tmp/
├── hostfile                    # MPI host specification
├── simple_rccl_test.sh        # Basic 2-GPU test
├── comprehensive_rccl_test.sh  # Full test suite
├── rccl_test_summary.sh       # Results summary
└── rccl-tests/                 # Compiled test binaries
    └── build/
        ├── all_reduce_perf     # Main test executable
        ├── all_gather_perf     # AllGather test
        └── ... (other tests)
```

## Expected Results

Successful execution should show:
- ✅ All network interfaces operational
- ✅ Low latency (2-10 μs) for small messages
- ✅ High bandwidth (1-3 GB/s) for large messages
- ✅ No data corruption (#wrong = 0)
- ✅ Successful scaling across all GPUs

## Support

For issues related to:
- **ROCm/RCCL**: Check ROCm documentation and GitHub issues
- **Network Configuration**: Verify netplan syntax and interface availability
- **MPI Problems**: Ensure OpenMPI is properly configured and SSH access works
- **Performance Issues**: Check GPU utilization, network bandwidth, and system resources

## License

These scripts are provided under the MIT License. RCCL and ROCm have their own respective licenses.