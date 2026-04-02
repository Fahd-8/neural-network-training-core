# PyTorch Neural Network Training — From OOP to AdamW

> A ground-up study of PyTorch neural network training — starting from Python OOP and building toward a complete training loop. No black boxes. Every line explained from first principles.

---

## What This Covers

Every concept is built from scratch — math first, then PyTorch, then application.

### 01 — OOP & nn.Module
- `__init__`, `self`, inheritance
- Why `super().__init__()` exists
- How `nn.Module` works as a parent class
- What you get for free by inheriting it

### 02 — Forward Pass
- `nn.Linear(in, out)` — the math behind it
- `x × w + b` from scratch in pure Python
- What `forward(self, x)` does and where `x` comes from
- Why you call `model(x)` not `model.forward(x)`

### 03 — Loss Functions
- What loss represents — how wrong the prediction is
- MSELoss — `(predicted - actual)²`
- CrossEntropyLoss — `-log(confidence)` for NLP
- Why loss can never drop for no reason

### 04 — Backpropagation
- What `loss.backward()` actually does
- Gradients — blaming each weight for the error
- Chain rule — `dy/dx = dy/du × du/dx`
- Why PyTorch tracks `grad_fn` on every tensor

### 05 — Optimizers
| Optimizer | What it does |
|---|---|
| Vanilla SGD | Blind steps, same size every time |
| SGD + Momentum | Builds velocity, reaches target faster |
| SGD + Nesterov | Momentum but looks ahead before stepping |
| Adam | Adaptive step size per weight, best default |
| AdamW | Adam + decoupled weight decay, use for transformers |

### 06 — Training Loop
```python
for epoch in range(epochs):
    optimizer.zero_grad()          # clear old gradients
    output = model(x)              # forward pass
    loss = criterion(output, y)    # how wrong
    loss.backward()                # gradients
    optimizer.step()               # fix weights
```

### 07 — Weight Decay & Regularization
- Why large weights cause overfitting
- How weight decay shrinks weights every step
- Why Adam's weight decay was broken
- How AdamW decouples and fixes it

### 08 — Production Transformer Training
- Layer-wise learning rates
- Warmup + linear decay scheduler
- Gradient clipping
- Full AdamW training loop

---

## The Core Insight

```
data → forward pass → loss → gradients → optimizer → weights change → loss changes
```

This cycle is called the **training loop.** Every number in it has a reason. Nothing changes randomly. Loss dropping means predictions are genuinely getting closer to the real answer.

---

## Stack
- Python 3.x
- PyTorch
- No abstractions — everything built from first principles

---

## Who This Is For

Anyone who wants to understand what's actually happening inside neural network training — not just copy paste code but trace every number back to its cause.
