# Reproducing the two-station network results

These commands assume `node0` is the local machine, `node1` is reachable through key-only SSH, and one rail is active. The hostnames and `192.168.200.1/30` and `.2/30` addresses are examples; substitute your own SSH aliases, interfaces, device identifiers, and isolated point-to-point subnet.

> **GB300 recovery safety:** Never run `nvidia-smi --gpu-reset` (including
> `nvidia-smi -r`), unload or reload NVIDIA modules, or perform PCI
> unbind/rescan. Generic reset, reload, or reboot suggestions retained in raw
> metadata are provenance, not instructions. If a driver is unhealthy, stop
> GPU work and coordinate a controlled host reboot with the operator.

Set the site-specific identifiers from a fresh inventory before running the
recipe. Determine GPU identifiers only while the driver is healthy, and verify
that the selected CUDA device is the GB300 rather than a display GPU. The same
CUDA index is assumed on both systems for the MPI example; if indices differ,
use a per-rank wrapper or another host-local selector.

```bash
# Define these values for your systems before continuing:
# export MANAGEMENT_IFACE=...
# export RAIL_INTERFACE=...
# export RDMA_HCA=...
# export GB300_CUDA_INDEX=...
# export GB300_PCI_BUS_ID=...
# export DATA_DIRECT_PCI_BUS_ID=...

: "${MANAGEMENT_IFACE:?Set the management interface used by MPI control traffic}"
: "${RAIL_INTERFACE:?Set the active point-to-point rail interface}"
: "${RDMA_HCA:?Set the RDMA HCA for that rail}"
: "${GB300_CUDA_INDEX:?Set the verified GB300 CUDA index on both hosts}"
: "${GB300_PCI_BUS_ID:?Set the GB300 PCI bus ID used by perftest}"
: "${DATA_DIRECT_PCI_BUS_ID:?Set the ConnectX-8 Data Direct PCI bus ID}"
```

## Persistent rail configuration

Use separate rails rather than a Linux bond so NCCL can observe and schedule each NIC path independently.

```bash
# node0
sudo nmcli con add type ethernet ifname "$RAIL_INTERFACE" con-name cx8-p0-node1 \
  ipv4.method manual ipv4.addresses 192.168.200.1/30 ipv4.never-default yes \
  ipv6.method disabled 802-3-ethernet.mtu 9000

# node1
sudo nmcli con add type ethernet ifname "$RAIL_INTERFACE" con-name cx8-p0-node0 \
  ipv4.method manual ipv4.addresses 192.168.200.2/30 ipv4.never-default yes \
  ipv6.method disabled 802-3-ethernet.mtu 9000
```

For an additional rail, choose its interface explicitly and use a separate
point-to-point subnet. Do not add a gateway or DNS server to either high-speed
profile.

## Enable GB300 Data Direct

Run on both systems:

```bash
sudo rdma_topo topo
sudo rdma_topo write-grub-acs
```

`write-grub-acs` is a reboot boundary. Stop here and have the operator schedule
and perform a controlled reboot of each system. Do not reboot either host
automatically. After the operator confirms that both systems have returned,
run on each system:

```bash
sudo rdma_topo check
```

The final check must say that DMA function `$DATA_DIRECT_PCI_BUS_ID` and GPU
`$GB300_PCI_BUS_ID` share an IOMMU group. This platform uses the separate Data
Direct function because Grace does not provide the conventional ATS route.

## Raw GPUDirect RDMA

Build perftest with CUDA support if `ib_write_bw --help` does not show `--use_data_direct`:

The distribution-owned binary can remain in `/usr/bin`; install the
CUDA-enabled build in `/usr/local/bin` and invoke that path explicitly below.

```bash
git clone https://github.com/linux-rdma/perftest.git
cd perftest
./autogen.sh
./configure --prefix=/usr/local CUDA_H_PATH=/usr/local/cuda/include/cuda.h --enable-cudart
make -j"$(nproc)"
sudo make install
sudo ldconfig
```

Server on `node1`:

```bash
/usr/local/bin/ib_write_bw -d "$RDMA_HCA" -x 3 -F --report_gbits -D 10 \
  --use_cuda_bus_id="$GB300_PCI_BUS_ID" --use_cuda_dmabuf --use_data_direct \
  -p 19004 --qp=4
```

Client on `node0`:

```bash
/usr/local/bin/ib_write_bw -d "$RDMA_HCA" -x 3 -F --report_gbits -D 10 \
  --use_cuda_bus_id="$GB300_PCI_BUS_ID" --use_cuda_dmabuf --use_data_direct \
  -p 19004 --qp=4 192.168.200.2
```

Add `--bidirectional --report-both` on both ends for the full-duplex test.

## Tuned NCCL all-reduce

NCCL 2.31.2 and nccl-tests 2.19.7 were built from their official NVIDIA repositories. The benchmark uses one MPI rank and one GB300 per system:

```bash
mpirun -np 2 --host node0:1,node1:1 \
  --bind-to none \
  --mca pml ob1 --mca btl self,vader,tcp \
  --mca btl_tcp_if_include "$MANAGEMENT_IFACE" \
  -x LD_LIBRARY_PATH=/opt/nccl-2.31.2/lib \
  -x CUDA_VISIBLE_DEVICES="$GB300_CUDA_INDEX" \
  -x NCCL_IB_HCA="$RDMA_HCA" \
  -x NCCL_SOCKET_IFNAME="$RAIL_INTERFACE" \
  -x NCCL_NET_GDR_LEVEL=SYS \
  -x NCCL_DMABUF_ENABLE=1 \
  -x NCCL_ALGO=RING -x NCCL_PROTO=SIMPLE \
  -x NCCL_MIN_NCHANNELS=8 -x NCCL_MAX_NCHANNELS=8 \
  -x NCCL_IB_QPS_PER_CONNECTION=4 -x NCCL_IB_SPLIT_DATA_ON_QPS=1 \
  /opt/nccl-tests-2.19.7/bin/all_reduce_perf \
  -b 512M -e 2G -f 2 -g 1 -w 5 -n 20
```

Expected 2 GiB result: approximately 48.72 GB/s bus bandwidth with zero errors. Set `NCCL_DEBUG=INFO` once when validating a new machine and confirm the log contains `Data Direct DMA Interface is detected` and `GDRDMA(PCI)`.

Regenerate the chart with `python3 render_chart.py`; the script requires Matplotlib.
