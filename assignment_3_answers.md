# CS336 Assignment 3 — Written Answers

## 2.1 Problem (chinchilla_isoflops): IsoFLOPs scaling laws

**(a)**

![compute_optimal_model.png](assets/compute_optimal_model.png)

By fitting linear function into minima of constant compute sweeps my optimal model sizes are:

- 9.114e+10 params for 1e23 FLOPs
- 2.981e+11 params for 1e24 FLOPs

**(b)**

![compute_optimal_data.png](assets/compute_optimal_data.png)

By approximating D ≈ C / 6N and fitting a linear function I received:

- 1.829e+11 tokens for 1e23 FLOPs
- 5.592e+11 tokens for 1e24 FLOPs

## 3 Problem (scaling_laws): Constructing scaling laws

Self-study setup: 3 A100-80GB hours for experiments, 12 for the final run. Recipe: DeepSeek-style (L11 p30), standard
parametrization, warmup-cosine.

### 3.0 Ground rules

- [x] What counts against the 3 h budget
  - Training-loop time only, as timed by `run.py` (excludes startup, data copy, compile)
  - Every run counts, including smoke tests and failed runs
  - Timed-out runs count their full `max_runtime_seconds`
  - Modal cost tracked separately
- [x] Compute definition
  - C = 6·N·D, N = non-embedding params = 12·n_layer·d_model², D = training tokens
- [x] Sweep axis
  - IsoFLOP sweeps at equal C, not equal wall-clock
  - Hardware efficiency handled once, at final-model scale (3.1, 3.6)

### 3.1 Time → compute

- [ ] Per-run overhead
- [ ] Throughput vs model size (and batch size)
- [ ] Throughput at and around the final N_opt
- [ ] C_target for 12 h

### 3.2 Fixed choices

- [x] Model shape rule: one family scaled by d_model
  - d_model: multiple of 64, min 256
  - n_layer = d_model / 64 (aspect ratio 64)
  - head_dim = 64, n_heads = d_model / 64
  - d_ff = 8/3 · d_model, rounded to a multiple of 64
- [x] Hyperparameters kept at handout defaults (only peak LR and batch size are swept, 3.3)
  - AdamW β1 0.9, β2 0.95, eps 1e-8, weight decay 0.01, grad clip 1.0
  - Warmup 5%, cosine to 10% of peak
  - bf16, untied embeddings, no attention bias, rope_theta 1e6

### 3.3 LR and batch size

- [ ] Sweep design
- [ ] Best LR vs C
- [ ] Best batch size vs C

### 3.4 IsoFLOPs

- [ ] Budgets and model sizes per budget
- [ ] N_opt(C), D_opt(C)

### 3.5 Loss prediction

- [ ] L_opt(C)

### 3.6 Final run

- [ ] Config: N, total_train_tokens, shape, LR, batch, max_runtime
- [ ] Predicted loss
- [ ] Actual loss

### Budget ledger

| Phase | Planned h | Used h |
|-------|-----------|--------|
| 3.1   |           |        |
| 3.3   |           |        |
| 3.4   |           |        |
| Final | 12        |        |
