# Self-Pruning Neural Network — Case Study Report

**Author:** Mohit Balachander | VIT-AP | 22MIC7042
**Dataset:** CIFAR-10
**Framework:** PyTorch

---

## 1. Why Does L1 Penalty on Sigmoid Gates Encourage Sparsity?

The sparsity objective is defined as the sum of all gate activations:


Σ sigmoid(gate_scores)


Each gate lies between 0 and 1 after the sigmoid function. Lower gate values reduce the contribution of the corresponding weight during the forward pass.

By adding this term to the training loss, the network is encouraged to keep only important gates active while pushing unnecessary ones toward zero.

The full objective becomes:


Total Loss = CrossEntropyLoss + λ × Σ sigmoid(gate_scores)


Where:

* **CrossEntropyLoss** rewards correct classification
* **λ (lambda)** controls pruning strength
* **Gate penalty** encourages sparsity

### Why L1 Works Better Than L2

#### L1 Penalty

The gradient remains approximately constant, so even small gates continue receiving downward pressure.

This causes many gates to shrink strongly toward zero.

#### L2 Penalty

The gradient becomes smaller as the gate value becomes smaller.

That means tiny gates stop shrinking efficiently and remain near-zero rather than effectively zero.

### Conclusion

L1 regularization is better suited for sparse models because it actively removes weak connections instead of merely reducing them.

---

## 2. Results — Lambda vs Accuracy vs Sparsity

| Lambda (λ) | Test Accuracy | Sparsity (%) |
| ---------- | ------------- | ------------ |
| 1e-06      | 55.70%        | 98.06%       |
| 1e-05      | 56.30%        | 99.91%       |
| 0.0001     | 56.34%        | 99.98%       |

---

## 3. Experimental Analysis

### λ = 1e-06

* Weak pruning pressure
* Highest number of surviving gates
* Still achieved **98.06% sparsity**
* Strong baseline accuracy

### λ = 1e-05

* Stronger sparsity pressure
* Nearly all redundant weights removed
* Accuracy improved to **56.30%**

### λ = 0.0001

* Most aggressive regularization among tested values
* Achieved **99.98% sparsity**
* Highest accuracy: **56.34%**

---

## 4. Key Findings

### Extreme Compression With Stable Accuracy

Even after removing nearly all effective weights, accuracy remained stable around **56%**.

This suggests that a very small subset of learned connections carried most of the predictive power.

### Pruning Did Not Harm Generalization

Instead of over-penalizing the model, sparsity regularization likely reduced noisy or unnecessary parameters.

### Best Overall Result


Lambda = 0.0001
Accuracy = 56.34%
Sparsity = 99.98%

This represents the best tradeoff achieved in the experiments.

---

## 5. Gate Value Distribution Plot

The plot below shows final gate values for the best-performing model:

```md
![Gate Distribution](gate_distribution.png)
```

### Interpretation

* A very large concentration of gates appears near **0**
* Almost no gates remain above the prune threshold of **0.50**
* Only a tiny fraction of connections remain strongly active

This confirms that the self-pruning mechanism worked successfully during training.

The model automatically identified which parameters were useful and suppressed the rest.

---

## 6. Threshold Design Choice

A prune threshold of:


0.50


was used for counting active vs pruned gates.

This is intuitive because:

* Gate < 0.50 → mostly inactive
* Gate > 0.50 → meaningfully active

Using this threshold gives a practical interpretation of whether a connection contributes significantly during inference.

---

## 7. Implementation Notes

## PrunableLinear Layer

Each linear layer contains:

* Standard learnable weights
* Learnable gate scores of identical shape

During forward pass:


effective_weight = weight × sigmoid(gate_scores)


Then the linear transformation is applied normally.

## Automatic Differentiation

Both weights and gate scores are optimized jointly using PyTorch autograd.

## Architecture Used


3072 → 512 → 256 → 128 → 10


With:

* ReLU activations
* BatchNorm in hidden layers
* Adam optimizer
* 10 epochs training
* CPU execution

---

## 8. Final Conclusion

This case study successfully demonstrates a trainable self-pruning neural network using differentiable gates.

The network learned to remove nearly all unnecessary parameters while preserving classification accuracy.

Final best result:


56.34% Accuracy at 99.98% Sparsity


This shows that structured sparsity can dramatically compress neural networks without major performance loss when implemented correctly.
