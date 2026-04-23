# Self-Pruning Neural Network
**Tredence Analytics — AI Engineering Intern Case Study**  
Mohit Balachander | VIT-AP | 22MIC7042

---

## What This Is

A feed-forward neural network that learns to prune its own weights **during training** — no post-hoc pruning step needed.

Each weight has a learnable gate (sigmoid output, range 0–1). An L1 penalty on all gate values forces most of them toward zero, effectively removing unnecessary connections on the fly.

## How to Run

```bash
pip install torch torchvision matplotlib
```

Download CIFAR-10 manually from https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz  
Extract into `data/cifar-10-python/` then:

```bash
python train.py
```

Runs 3 experiments (λ = 0.0001, 0.001, 0.01), prints results, saves `gate_distribution.png`.

## Results

| Lambda (λ) | Test Accuracy | Sparsity (%) |
|:---:|:---:|:---:|
| 0.0001 | 55.37% | 93.19% |
| 0.001  | 55.70% | 99.95% |
| 0.01   | 53.29% | 100.00% |

At 99.95% sparsity the network retains ~55.7% accuracy — almost all weights were redundant.

## Key Design Decision

The sparsity loss must be computed **inside the computation graph** (no `.detach()`).  
If gates are detached before summing, gradients never reach `gate_scores` and pruning doesn't happen.

## Files

- `train.py` — full implementation: PrunableLinear layer, network, training loop, evaluation
- `report.md` — analysis of L1 sparsity, results table, plot interpretation
- `gate_distribution.png` — histogram of gate values after training