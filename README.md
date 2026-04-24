# Self-Pruning Neural Network
**Tredence Analytics — AI Engineering Intern Case Study**  
Mohit Balachander | VIT-AP | 22MIC7042

---

## What This Is

A feed-forward neural network that learns to prune its own weights **during training** — no post-hoc pruning step needed.

Each weight has a learnable gate (sigmoid output, range 0–1). An L1 penalty on all gate
values forces most of them toward zero during training, effectively removing unnecessary
connections on the fly.

The network goes through two distinct phases:
1. **Epochs 1–4:** Gates start open (≈0.88), network learns useful representations first
2. **Epoch 5 onward:** Sparsity penalty wins — gates collapse, pruning kicks in sharply

---

## How to Run

```bash
pip install torch torchvision matplotlib
```

Download CIFAR-10 manually from:  
https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

Extract into `data/cifar-10-python/` then:

```bash
python train.py
```

Runs 3 lambda experiments, prints per-epoch results, saves `gate_distribution.png`.

---

## Results

| Lambda (λ) | Test Accuracy | Sparsity Level (%) |
|:----------:|:-------------:|:------------------:|
| 0.0001     | 51.97%        | 97.80%             |
| 0.001      | 50.93%        | 99.80%             |
| 0.01       | 49.82%        | 99.99%             |

At λ=0.001: **99.80% of weights pruned** while retaining **50.93% accuracy** on CIFAR-10.  
Accuracy visibly drops as λ increases — the sparsity-accuracy tradeoff is clear and measurable.

---

## Key Design Decisions

**Gate initialisation at +2.0**  
`gate_scores` initialise at 2.0, so `sigmoid(2.0) ≈ 0.88` — gates start mostly open.
This lets the network learn before pruning, producing cleaner two-phase dynamics.
Initialising at 0 causes gates to collapse from epoch 1 before any learning happens.

**In-graph sparsity loss**  
The sparsity loss must be computed without `.detach()` on the gate values.
If detached, gradients never reach `gate_scores` and gates don't move.

**Smaller architecture (3072→64→32→16→10)**  
Fewer total gates means each gate matters more for accuracy. The network genuinely
fights to keep important connections alive, producing interpretable pruning behaviour.

---

## Files

| File | Description |
|---|---|
| `train.py` | Full implementation: PrunableLinear, network, training loop, evaluation |
| `report.md` | L1 sparsity analysis, results table, plot interpretation, design decisions |
| `gate_distribution.png` | Histogram of final gate values for best model |
| `requirements.txt` | Exact package versions for reproducibility |