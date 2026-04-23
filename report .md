# Self-Pruning Neural Network — Case Study Report
**Author:** Mohit Balachander  
**Dataset:** CIFAR-10  
**Framework:** PyTorch  

---

## 1. Why Does L1 Penalty on Sigmoid Gates Encourage Sparsity?

The sparsity loss is defined as the **L1 norm of all gate values** — that is, the sum of `sigmoid(gate_scores)` across every weight in the network.

Since sigmoid outputs are always positive, minimizing this sum means driving as many gate values as possible toward **exactly zero**.

The key reason L1 works here (and L2 does not) lies in their gradient behaviour:

- **L1 gradient** with respect to a gate value `g` is constant: `+λ` (sign of g, which is always +1 since gates > 0). This means every gate, no matter how small, receives the same constant downward pressure. Gates are pushed all the way to zero and stay there.

- **L2 gradient** is proportional to `2λg`. As a gate gets smaller, the gradient shrinks too — so the push toward zero weakens and gates settle near-zero but never reach it. L2 encourages *small* weights, not *absent* ones.

This is why L1 regularization is the standard choice for inducing true sparsity in machine learning. In our setup, the total loss is:

```
Total Loss = CrossEntropyLoss(predictions, labels) + λ × Σ sigmoid(gate_scores)
```

The network must constantly balance two competing objectives: classify correctly (minimize CrossEntropy) while keeping as few gates open as possible (minimize L1 sparsity term). Weights that do not contribute meaningfully to classification get their gates pushed to zero — effectively pruned.

---

## 2. Results — Lambda vs Accuracy vs Sparsity

| Lambda (λ) | Test Accuracy | Sparsity Level (%) |
|:---:|:---:|:---:|
| 0.0001 | 55.94% | 93.24% |
| 0.001  | 55.68% | 99.94% |
| 0.01   | 54.82% | 100.00% |

**Key observations:**

- Even at 93% sparsity (λ = 0.0001), the network retains ~55.9% accuracy — meaning the vast majority of weights were genuinely redundant.
- At λ = 0.001, the network prunes 99.94% of weights while losing less than 0.3% accuracy. This is the sweet spot — extreme compression with minimal accuracy cost.
- At λ = 0.01, full sparsity is reached but accuracy drops slightly as the penalty becomes aggressive enough to prune some genuinely useful connections.
- Accuracy remains remarkably stable across all three λ values (~55%), confirming that the network successfully identifies and preserves the small subset of weights that matter most.

---

## 3. Gate Value Distribution Plot

The plot below shows the distribution of all gate values after training with λ = 0.0001 (the best accuracy model).

![Gate Distribution](gate_distribution.png)

**Interpretation:**  
The large spike near zero confirms that the vast majority of weights have been pruned — their gates were driven close to 0 by the L1 penalty. The long tail extending toward 0.3 represents the surviving connections that the network determined were essential for classification. This bimodal-like pattern (spike at 0, small population of active gates) is exactly the expected signature of successful learned sparsity.

---

## 4. Implementation Notes

**PrunableLinear Layer:**  
Each layer maintains a `gate_scores` tensor of the same shape as `weight`, registered as an `nn.Parameter`. During the forward pass, `sigmoid(gate_scores)` produces gates in (0, 1), which are multiplied element-wise with the weights before the linear operation. Both `weight` and `gate_scores` receive gradients automatically via PyTorch autograd.

**Critical design decision — in-graph sparsity loss:**  
The sparsity loss must be computed *inside* the computation graph (without `.detach()`) so that gradients flow back to `gate_scores` during `loss.backward()`. If gate values are detached before summing, the sparsity penalty has no effect on gate training.

**Network architecture:**  
Four PrunableLinear layers: 3072 → 512 → 256 → 128 → 10, with BatchNorm and ReLU activations. Trained with Adam (lr=1e-3) for 10 epochs per experiment.