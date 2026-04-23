# train.py
# Mohit Balachander | VIT-AP | 22MIC7042
# Tredence Analytics — AI Engineering Intern Case Study
# Self-Pruning Neural Network on CIFAR-10
#
# Approach: attach a learnable sigmoid gate to every weight.
# An L1 penalty on those gates forces most of them to zero
# during training — no post-hoc pruning step needed.

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Part 1 — PrunableLinear
# ---------------------------------------------------------------------------
# The standard nn.Linear computes: output = input @ weight.T + bias
# We add a gate tensor of the same shape as weight.
# gate = sigmoid(gate_scores)  →  always in (0, 1)
# effective_weight = weight * gate
# When gate → 0, that weight stops contributing — it's pruned.
#
# Both weight and gate_scores are nn.Parameters, so autograd tracks them
# both. The key design choice: get_sparsity_loss() must NOT call .detach()
# on the gates — if it does, no gradient reaches gate_scores and pruning
# never happens. Learned this the hard way during debugging.

class PrunableLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()
        # weights initialised small so gates dominate early
        self.weight      = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias        = nn.Parameter(torch.zeros(out_features))
        # gate_scores start at 0 → sigmoid(0) = 0.5, gates half-open
        self.gate_scores = nn.Parameter(torch.zeros(out_features, in_features))

    def forward(self, x):
        gates          = torch.sigmoid(self.gate_scores)   # (out, in) in (0,1)
        pruned_weights = self.weight * gates               # element-wise mask
        return nn.functional.linear(x, pruned_weights, self.bias)

    def sparsity_loss(self):
        # sum of gates — stays IN the computation graph
        # so d(loss)/d(gate_scores) is non-zero and gates actually move
        return torch.sigmoid(self.gate_scores).sum()

    def gates_numpy(self):
        # detach only when we need values for analysis, never for the loss
        return torch.sigmoid(self.gate_scores).detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Part 2 — Network
# ---------------------------------------------------------------------------
# Four prunable layers: 3072 → 512 → 256 → 128 → 10
# CIFAR-10 images are 32x32x3 = 3072 floats when flattened.
# BatchNorm after the first two layers stabilises training — without it
# the loss was noisy and accuracy topped out around 48%.

class SelfPruningNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1  = PrunableLinear(3072, 512)
        self.bn1  = nn.BatchNorm1d(512)
        self.fc2  = PrunableLinear(512,  256)
        self.bn2  = nn.BatchNorm1d(256)
        self.fc3  = PrunableLinear(256,  128)
        self.fc4  = PrunableLinear(128,  10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)            # flatten spatial dims
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.relu(self.fc3(x))
        return self.fc4(x)

    def total_sparsity_loss(self):
        # collect in-graph gate sums from every PrunableLinear
        total = sum(
            m.sparsity_loss()
            for m in self.modules()
            if isinstance(m, PrunableLinear)
        )
        return total

    def all_gates(self):
        import numpy as np
        parts = [
            m.gates_numpy().flatten()
            for m in self.modules()
            if isinstance(m, PrunableLinear)
        ]
        combined = np.concatenate(parts)
        return torch.tensor(combined)


# ---------------------------------------------------------------------------
# Part 3 — Data
# ---------------------------------------------------------------------------

def get_loaders(batch_size=128):
    # standard CIFAR-10 mean/std normalisation
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])
    train = torchvision.datasets.CIFAR10(
        root='./data/cifar-10-python', train=True,  download=False, transform=tf)
    test  = torchvision.datasets.CIFAR10(
        root='./data/cifar-10-python', train=False, download=False, transform=tf)

    return (
        torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True,  num_workers=0),
        torch.utils.data.DataLoader(test,  batch_size=batch_size, shuffle=False, num_workers=0),
    )


# ---------------------------------------------------------------------------
# Part 4 — Training loop
# ---------------------------------------------------------------------------
# Total Loss = CrossEntropy + lambda * sum(gates)
#
# CrossEntropy pulls gate_scores toward whatever helps classification.
# The lambda term pulls every gate toward 0.
# The network reaches a per-weight equilibrium: gates that improve
# accuracy enough to justify the penalty survive; the rest go to zero.

def train_epoch(model, loader, optimizer, criterion, lam, device):
    model.train()
    running = 0.0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        logits   = model(imgs)
        cls_loss = criterion(logits, labels)
        spr_loss = model.total_sparsity_loss()   # in-graph
        loss     = cls_loss + lam * spr_loss

        loss.backward()
        optimizer.step()
        running += loss.item()

    return running / len(loader)


# ---------------------------------------------------------------------------
# Part 5 — Evaluation helpers
# ---------------------------------------------------------------------------

def accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds        = model(imgs).argmax(dim=1)
            correct     += (preds == labels).sum().item()
            total       += labels.size(0)
    return 100 * correct / total


def sparsity(model, threshold=0.1):
    # % of gates below threshold -> weight is considered pruned
    gates  = model.all_gates()
    pruned = (gates < threshold).sum().item()
    return 100 * pruned / gates.numel()


# ---------------------------------------------------------------------------
# Part 6 — Single experiment
# ---------------------------------------------------------------------------

def run(lam, train_loader, test_loader, device, epochs=10):
    print(f"\n{'─'*54}")
    print(f"  Mohit Balachander | experiment  lambda = {lam}")
    print(f"{'─'*54}")

    model     = SelfPruningNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for ep in range(1, epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, lam, device)
        acc  = accuracy(model, test_loader, device)
        spr  = sparsity(model)
        print(f"  ep {ep:02d}/{epochs}  loss={loss:8.2f}  acc={acc:.2f}%  sparsity={spr:.1f}%")

    final_acc = accuracy(model, test_loader, device)
    final_spr = sparsity(model)
    print(f"\n  result -> acc={final_acc:.2f}%   sparsity={final_spr:.2f}%")
    return model, final_acc, final_spr


# ---------------------------------------------------------------------------
# Part 7 — Plot
# ---------------------------------------------------------------------------

def plot(model, lam):
    # Mohit Balachander — gate distribution for best model
    gates = model.all_gates().numpy()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(gates, bins=100, color='steelblue', edgecolor='black', alpha=0.82)
    ax.axvline(0.1, color='crimson', linestyle='--', linewidth=1.6,
               label='prune threshold = 0.10')
    ax.set_title(f'Gate Value Distribution  (lambda = {lam})\n'
                 f'Mohit Balachander — Tredence Case Study', fontsize=13)
    ax.set_xlabel('Gate value  [0 = pruned  →  1 = fully active]', fontsize=11)
    ax.set_ylabel('Weight count', fontsize=11)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig('gate_distribution.png', dpi=150)
    plt.close()
    print('  plot saved -> gate_distribution.png')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # Mohit Balachander | 22MIC7042 | VIT-AP
    device = torch.device('cpu')
    print(f'device: {device}')

    train_loader, test_loader = get_loaders()

    # three lambda values to demonstrate the sparsity-accuracy tradeoff
    lambdas = [1e-4, 1e-3, 1e-2]

    results    = []
    best_model = None
    best_lam   = None
    best_acc   = -1.0

    for lam in lambdas:
        model, acc, spr = run(lam, train_loader, test_loader, device, epochs=10)
        results.append((lam, acc, spr))
        if acc > best_acc:
            best_acc, best_model, best_lam = acc, model, lam

    # summary
    print('\n\n' + '='*54)
    print(f"  {'Lambda':<12}  {'Accuracy':>12}  {'Sparsity':>12}")
    print('='*54)
    for lam, acc, spr in results:
        print(f"  {lam:<12}  {acc:>11.2f}%  {spr:>11.2f}%")
    print('='*54)

    plot(best_model, best_lam)

    print('\ndone.')
    print('Mohit Balachander | VIT-AP | 22MIC7042')