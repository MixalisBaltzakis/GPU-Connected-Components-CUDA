# GPU-Connected-Components-CUDA

This repository contains a high-performance CUDA C++ implementation of the Connected Components Label Propagation algorithm, designed to process large-scale undirected graphs on the GPU. The implementation explores and benchmarks different levels of hardware parallelism to maximize GPU utilization.

##  Architecture & Parallelization Strategies

The graph data is represented using the Compressed Sparse Row (CSR) / Compressed Sparse Column (CSC) format. To evaluate the performance trade-offs based on graph topology, three distinct parallelization strategies were developed:

1. **Thread-per-Row:** Assigns a single GPU thread to process all neighbors of a specific node.
2. **Warp-per-Row:** Assigns an entire warp (32 threads) to collaboratively read a node's neighbors and update its label. It utilizes intra-warp reductions at the hardware register level to efficiently find local minimums.
3. **Block-per-Row:** Dedicates a full thread block (e.g., 128 threads) to a single node. It leverages shared memory (`__shared__`) for fast intra-block communication and multi-stage reduction (warp-level followed by block-level) to determine the global minimum.

##  Hardware-Level Optimizations

To ensure the GPU operates at peak capacity, the following optimizations were implemented:
* **Grid-Stride Loops:** Deployed in the Block-per-Row strategy, starting a dynamic number of blocks based on the Streaming Multiprocessors (SMs). This enables zero-cost context switching to hide latency.
* **Memory Coalescing:** Warp and Block strategies ensure that adjacent threads read contiguous memory addresses, avoiding non-coalesced memory accesses and drastically reducing the required memory transactions (128-byte packets).

##  Performance Insights

Profiling revealed that the optimal strategy heavily depends on the graph's average degree:
* For graphs with a very low average degree (e.g., `kmer_A2a`), the **Thread-per-Row** approach is the fastest, as using entire warps or blocks for 2-3 neighbors leaves most threads idle.
* For dense graphs or mesh structures (e.g., `Queen_4147`, `rgg_n_2_24_s0`), the **Warp-per-Row** and **Block-per-Row** strategies are vastly superior. They effectively distribute the heavy workload, hide latency, and prevent memory bandwidth bottlenecks caused by non-coalesced accesses.

##  How to Run

The entire pipeline—from data ingestion to CUDA kernel execution and benchmarking—is encapsulated within a Jupyter Notebook. It is highly recommended to run this project in **Google Colab** (or any Jupyter environment with an NVIDIA GPU).

1. Clone this repository.
2. Upload the `.ipynb` notebook to Google Colab.
3. Ensure the Runtime is set to use a GPU (Runtime -> Change runtime type -> GPU).
4. Run all cells sequentially. The notebook will handle downloading the `.mat` datasets, converting them, and compiling/executing the CUDA C++ code automatically.
