# CFLAG-BD: Blockchain-Based Clustered Federated Learning with Deduplication of Model Weights

## This repository contains the source code, models, experiments, and documentation for the paper:

> **Blockchain-Based Clustered Federated Learning with Deduplication of Model Weights**  
> Aliyu Garba, Universiti Teknologi PETRONAS  
> Manuscript ID: BCRA-D-24-00448 (Under Review, *Blockchain: Research and Applications*)

---

## Overview

CFLAG-BD is a privacy-preserving and communication-efficient federated learning framework that:
- Dynamically clusters clients based on multi-dimensional criteria (data similarity, processing power, reliability, and architecture).
- Integrates a blockchain-based deduplication mechanism to prevent redundant model updates.
- Supports deployment in heterogeneous edge environments (e.g., IoT, mobile, healthcare).
- Achieves scalability up to 10,000 clients with minimal blockchain overhead and high model convergence.

---

## Requirements

- Python 3.8+
- VS Code or any compatible IDE (e.g., PyCharm, Jupyter)
- PyTorch ≥ 1.10
- Packages:
  - `numpy`
  - `scikit-learn`
  - `matplotlib`
  - `tqdm`
  - `seaborn`
  - `flask` *(for simulated blockchain)*
  - `networkx`
- Optional: 
  - Docker + Hyperledger Fabric (for actual blockchain deployment)
  - CUDA-enabled GPU for large-scale simulation

> Use the provided [`requirements.txt`](./requirements.txt) to install dependencies:
```bash
pip install -r requirements.txt
```
## Project Structure
```bash
CFLAG-BD/
├── main_experiment.py               # Main orchestration for training and evaluation
├── cflag_bd_algorithm.py           # Core implementation of CFLAG-BD logic
├── memory_efficient_cflagbd.py     # Memory-optimized federated simulation
├── baseline_algorithms.py          # Implementations of CQFL, IFCA, FedAvg, FedProx, LayerCFL
├── model_architectures.py          # MLP, CNN, LSTM models
├── lazy_dataset.py                 # Dataset wrapper with partitioning for federated use
├── enhanced_client.py              # Federated client logic with clustering hooks
├── learning_rate_scheduler.py      # Adaptive LR strategy
├── hardware_profiler.py            # Emulates heterogeneous hardware benchmarks
├── multi_dataset_support.py        # Switching between CIFAR, FMNIST, HAR datasets
├── monitored_execution.py          # Resource/time profiling
├── analyze_convergence.py          # Plot and log convergence metrics
├── results/                        # Folder with visualizations and logs
└── README.md
```
## Running Experiments
### Simulate Federated Learning with CFLAG-BD
```bash
python main_experiment.py --dataset cifar10 --algorithm cflag-bd --clients 1000
```
## Parameters
```
[-h] [--results_dir RESULTS_DIR] [--save_plots]
                          [--experiment {single,comparative,ablation,scalability,all}] [--dataset DATASET]
                          [--method METHOD] [--num_clients NUM_CLIENTS] [--num_rounds NUM_ROUNDS]
                          [--local_epochs LOCAL_EPOCHS] [--batch_size BATCH_SIZE] [--client_fraction CLIENT_FRACTION]
                          [--num_clusters NUM_CLUSTERS] [--dedup_threshold DEDUP_THRESHOLD] [--fedprox_mu FEDPROX_MU]
                          [--quantization_bits QUANTIZATION_BITS] [--non_iid] [--use_blockchain] [--dynamic_lr]
                          [--lr_schedule {step,exponential,cosine,adaptive,cyclic}]
```
## Optional arguments:
    --clusters: Number of client clusters (e.g., 5)
    --dedup: Enable deduplication logic
    --rounds: Communication rounds (default: 50)

## Run Baselines              
```bash
  python experiment_runner.py --baseline fedprox
```
## Scalability Testing (10,000 clients)
```bash
python memory_efficient_cflagbd.py --clients 10000
```
## Evaluation Metrics
- Classification accuracy (global and per cluster)
- Communication cost (per round & cumulative)
- Convergence speed (accuracy vs. round)
- Storage footprint per client and block
- Blockchain overhead (block time, consensus time)
- Statistical significance (p-values, Cohen’s d)
- Scalability (training time across 100–10,000 clients) 
## Datasets Used
- [`Fashion-MNIST`](https://github.com/zalandoresearch/fashion-mnist)
- [`CIFAR-10`](https://www.cs.toronto.edu/~kriz/cifar.html)
  Use the **--dataset** flag to specify which one to load.
## Citation
If you use this work, please cite it as:
### IEEE Citation:
```bash
Aliyu Garba, Norshakirah Aziz, Hitham Alhussian, Abdullahi Abubakar Imam, Ridwan Salahudden, Shamsudeen Adamu, and Yahaya Saidu, "Blockchain-Based Clustered Federated Learning with Deduplication of Model Weights," IEEE Conference Paper, 2025. [To appear].
```
### APA Citation:
```bash
Garba, A., Aziz, N., Alhussian, H., Imam, A. A., Salahudden, R., Adamu, S., & Saidu, Y. (2025). Blockchain-Based Clustered Federated Learning with Deduplication of Model Weights. IEEE. [To appear].
```

## Contact

For collaborations, questions, or contributions:

Aliyu Garba
- engrabusadik@gmail.com
- https://github.com/engrabu/CFLAG-DB/
