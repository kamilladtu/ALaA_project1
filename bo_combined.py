import os
import numpy as np
import matplotlib.pyplot as plt
import skopt
from skopt import gp_minimize

# ─────────────────────────────────────────────────────────────────────────────
# RESULTS FOLDER
# ─────────────────────────────────────────────────────────────────────────────
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
print(f"Saving plots to ./{RESULTS_DIR}/")

from skopt.space import Integer, Real
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

np.random.seed(32)
np.int = int  # numpy deprecation workaround required by skopt


# ─────────────────────────────────────────────────────────────────────────────
# SEARCH SPACE  (single source of truth)
# ─────────────────────────────────────────────────────────────────────────────
FIXED_LR = 1e-3

SEARCH_SPACE = {
    "batch_size":   (16,  128),   # large batches hurt on small data
    "num_filters":  (4,   32),    # 4 filters is genuinely weak → real bottleneck
    "dense_units":  (16,  128),   # 16 units is very small for a 10-class head
    "dropout_rate": (0.2, 0.7),   # 0.7 kills too much signal
}

# skopt space (used by all BO runs)
space = [
    Integer(*SEARCH_SPACE["batch_size"],   name="batch_size"),
    Integer(*SEARCH_SPACE["num_filters"],  name="num_filters"),
    Integer(*SEARCH_SPACE["dense_units"],  name="dense_units"),
    Real(   *SEARCH_SPACE["dropout_rate"], name="dropout_rate"),
]

# Random sampler (used by Random Search and BO seed points)
def sample_random_params(rng: np.random.Generator, n: int):
    batch_sizes   = rng.integers(*SEARCH_SPACE["batch_size"],   size=n)
    num_filters   = rng.integers(*SEARCH_SPACE["num_filters"],  size=n)
    dense_units   = rng.integers(*SEARCH_SPACE["dense_units"],  size=n)
    dropout_rates = rng.uniform( *SEARCH_SPACE["dropout_rate"], size=n)
    return batch_sizes, num_filters, dense_units, dropout_rates


# ── 1. Load & preprocess MNIST ───────────────────────────────────────────────
(train_X, train_y), (test_X, test_y) = mnist.load_data()

train_X = train_X.reshape(-1, 28, 28, 1).astype('float32') / 255.0
test_X  = test_X.reshape(-1,  28, 28, 1).astype('float32') / 255.0

train_y_cat = to_categorical(train_y, 10)
test_y_cat  = to_categorical(test_y,  10)

# Constrained subset: bad configs score ~82%, good configs ~97%
# → hyperparameter choices actually matter
N_TRAIN, N_TEST = 4_000, 2_000
X_tr, y_tr = train_X[:N_TRAIN], train_y_cat[:N_TRAIN]
X_te, y_te = test_X[:N_TEST],   test_y_cat[:N_TEST]

# ── Proper three-way split (Script 2's key methodological fix) ───────────────
# Tuning uses validation accuracy only; test set is touched exactly once at the end.
X_train, X_val, y_train, y_val = train_test_split(
    X_tr, y_tr,
    test_size=0.2,
    random_state=32,
    stratify=train_y[:N_TRAIN],
)

print("Data splits:")
print(f"  Train : {X_train.shape[0]} samples")
print(f"  Val   : {X_val.shape[0]}  samples  (used as tuning signal)")
print(f"  Test  : {X_te.shape[0]}   samples  (touched only for final evaluation)")


# ── 2. Model builder ──────────────────────────────────────────────────────────
# Single conv block kept shallow on purpose so num_filters is the capacity
# bottleneck — its effect is clearly visible in the surrogate heatmap.

def build_model(num_filters: int, dense_units: int, dropout_rate: float) -> Sequential:
    model = Sequential([
        Input(shape=(28, 28, 1)),
        Conv2D(int(num_filters), (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(int(dense_units), activation='relu'),
        Dropout(dropout_rate),
        Dense(10, activation='softmax'),
    ])
    model.compile(
        optimizer=Adam(learning_rate=FIXED_LR),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def build_and_evaluate(
    batch_size: int, num_filters: int, dense_units: int, dropout_rate: float,
    epochs: int = 3,
) -> float:
    """Train on X_train, return VALIDATION accuracy (never touches test set)."""
    model = build_model(num_filters, dense_units, dropout_rate)
    early_stop = EarlyStopping(monitor='val_accuracy', patience=2,
                               restore_best_weights=True)
    model.fit(
        X_train, y_train,
        batch_size=int(batch_size),
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=[early_stop],
        verbose=0,
    )
    _, val_acc = model.evaluate(X_val, y_val, verbose=0)
    return float(val_acc)


def train_final_and_test(
    batch_size: int, num_filters: int, dense_units: int, dropout_rate: float,
    epochs: int = 5,
) -> float:
    """Retrain on train+val combined, evaluate ONCE on held-out test set."""
    X_final = np.concatenate([X_train, X_val], axis=0)
    y_final = np.concatenate([y_train, y_val], axis=0)
    model = build_model(num_filters, dense_units, dropout_rate)
    model.fit(X_final, y_final, batch_size=int(batch_size),
              epochs=epochs, verbose=0)
    _, test_acc = model.evaluate(X_te, y_te, verbose=0)
    return float(test_acc)


# ── 3. Random Search baseline ─────────────────────────────────────────────────
print("\n── Random Search (20 iterations) ─────────────────────────────────────")

N_ITER  = 20
N_INIT  = 5    # shared warm-start points for all BO runs
rng = np.random.default_rng(32)

rand_bs, rand_nf, rand_du, rand_dr = sample_random_params(rng, N_ITER)

current_best_val    = 0.0
best_random_params  = None
max_val_random      = []
random_all_vals     = []

for i in range(N_ITER):
    bs, nf, du, dr = int(rand_bs[i]), int(rand_nf[i]), int(rand_du[i]), float(rand_dr[i])
    val_acc = build_and_evaluate(bs, nf, du, dr)
    random_all_vals.append(val_acc)

    if val_acc > current_best_val:
        current_best_val   = val_acc
        best_random_params = (bs, nf, du, dr)

    max_val_random.append(current_best_val)
    print(f"  Iter {i+1:>2d}/{N_ITER} | batch={bs:>4d} | filters={nf:>3d} | "
          f"dense={du:>4d} | dropout={dr:.2f} | val={val_acc:.4f} | best={current_best_val:.4f}")


# ── 4. Shared BO warm-start (fair baseline for all acquisition functions) ─────
# Compute y0 ONCE from the first N_INIT random search points and reuse.
# This guarantees all BO variants start from identical GP priors.
print(f"\n── Computing shared y0 for {N_INIT} warm-start points ────────────────")

x0 = [[int(rand_bs[i]), int(rand_nf[i]), int(rand_du[i]), float(rand_dr[i])]
      for i in range(N_INIT)]
y0 = [-build_and_evaluate(*x) for x in x0]

for i, (x, y) in enumerate(zip(x0, y0), 1):
    print(f"  init {i}/{N_INIT} | x={x} | val_acc={-y:.4f}")


# ── 5. Bayesian Optimisation: EI vs PI vs LCB ─────────────────────────────────
print("\n── Bayesian Optimisation: EI vs PI vs LCB ─────────────────────────────")

N_BO_CALLS = N_ITER - N_INIT   # remaining guided evaluations

def run_bo(acq_func: str, xi=None, kappa=None):
    it = {"i": 1}
    bo_accs = []

    def objective(x):
        val_acc = build_and_evaluate(*x)
        bo_accs.append(val_acc)
        print(f"  [{acq_func}] iter {it['i']:>2d}/{N_BO_CALLS} | "
              f"batch={x[0]:>4d} | filters={x[1]:>3d} | dense={x[2]:>4d} | "
              f"dropout={x[3]:.2f} | val={val_acc:.4f}")
        it["i"] += 1
        return -val_acc

    kwargs = dict(
        func=objective,
        dimensions=space,
        n_calls=N_BO_CALLS,
        x0=x0, y0=y0,
        acq_func=acq_func,
        random_state=32,
        noise=0.01**2,
        n_initial_points=0,   # we supply x0/y0; no extra random init needed
    )
    if acq_func in ("EI", "PI") and xi is not None:
        kwargs["xi"] = xi
    if acq_func == "LCB" and kappa is not None:
        kwargs["kappa"] = kappa

    result = gp_minimize(**kwargs)
    return result, bo_accs

XI_VALUE    = 0.1
KAPPA_VALUE = 2.0

opt_EI,  accs_EI  = run_bo("EI",  xi=XI_VALUE)
opt_PI,  accs_PI  = run_bo("PI",  xi=XI_VALUE)
opt_LCB, accs_LCB = run_bo("LCB", kappa=KAPPA_VALUE)


# ── 6. Best val configs → final test evaluation ───────────────────────────────
print("\n── Final Test Evaluation (retrain on train+val) ─────────────────────")

def best_params(opt):
    return opt.x[0], opt.x[1], opt.x[2], opt.x[3]

configs = {
    "Random Search": best_random_params,
    "BO EI":         best_params(opt_EI),
    "BO PI":         best_params(opt_PI),
    "BO LCB":        best_params(opt_LCB),
}

test_results = {}
for name, params in configs.items():
    test_acc = train_final_and_test(*params)
    test_results[name] = test_acc
    print(f"  {name:<16s} | params={params} | test_acc={test_acc:.4f}")


# ── 7. Build convergence curves (full 20 iterations) ─────────────────────────
# Prepend the shared y0 values so curves start from iteration 1

def full_curve(opt):
    # opt.func_vals already contains ALL evaluations including x0/y0
    all_accs = -np.array(opt.func_vals)
    return np.maximum.accumulate(all_accs)

curve_EI  = full_curve(opt_EI)
curve_PI  = full_curve(opt_PI)
curve_LCB = full_curve(opt_LCB)
xs = np.arange(1, len(curve_EI) + 1)  # dynamic — skopt may include x0 in func_vals


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

# ── Plot 1: Convergence — Random Search vs all BO variants ───────────────────
plt.figure(figsize=(9, 5))
plt.plot(xs, max_val_random[:len(xs)], 'o-', color='crimson',   lw=2, label='Random Search')
plt.plot(xs, curve_EI,       'o-', color='steelblue', lw=2, label=f'BO EI  (xi={XI_VALUE})')
plt.plot(xs, curve_PI,       's-', color='darkorange', lw=2, label=f'BO PI  (xi={XI_VALUE})')
plt.plot(xs, curve_LCB,      '^-', color='seagreen',  lw=2, label=f'BO LCB (kappa={KAPPA_VALUE})')
plt.axvline(x=N_INIT, color='grey', linestyle='--', alpha=0.6, label=f'End of shared init (n={N_INIT})')
plt.xlabel('Iterations')
plt.ylabel('Best Validation Accuracy (cumulative)')
plt.title('Convergence: Random Search vs Bayesian Optimisation (EI / PI / LCB)\n'
          'Optimising on validation accuracy | Fixed LR=1e-3')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'bo_convergence.png'), dpi=150)
plt.show()
print("Saved: bo_convergence.png")


# ── Plot 2: Per-iteration scatter (raw val accuracy) ─────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(range(1, N_ITER+1), random_all_vals,
                color='crimson', alpha=0.7, s=60, label='Random Search')
for accs, label, color in [
    (accs_EI,  'BO EI',  'steelblue'),
    (accs_PI,  'BO PI',  'darkorange'),
    (accs_LCB, 'BO LCB', 'seagreen'),
]:
    # guided evals start after N_INIT
    xs_bo = range(N_INIT+1, N_ITER+1)
    axes[0].scatter(xs_bo, accs, alpha=0.7, s=60, label=label, color=color)

axes[0].axvline(x=N_INIT, color='grey', linestyle='--', alpha=0.5)
axes[0].set_xlabel('Iteration')
axes[0].set_ylabel('Validation Accuracy')
axes[0].set_title('Per-Iteration Validation Accuracy')
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

# Final test accuracy bar chart
names = list(test_results.keys())
vals  = list(test_results.values())
colors = ['crimson', 'steelblue', 'darkorange', 'seagreen']
bars = axes[1].bar(names, vals, color=colors, alpha=0.8, edgecolor='black')
axes[1].set_ylim(min(vals) - 0.02, min(max(vals) + 0.02, 1.0))
axes[1].set_ylabel('Test Accuracy')
axes[1].set_title('Final Test Accuracy\n(retrained on train+val, evaluated once on test)')
axes[1].grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, vals):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'bo_scatter_and_test.png'), dpi=150)
plt.show()
print("Saved: bo_scatter_and_test.png")


# ── Plot 3: GP Surrogate Heatmap (num_filters × dense_units, EI model) ────────
# Slice the 4D surrogate across the two main architectural axes,
# fixing batch_size and dropout_rate at the EI-optimal values.
gp_model = opt_EI.models[-1]
opt_bs   = opt_EI.x[0]
opt_dr   = opt_EI.x[3]

nf_vals = np.linspace(*SEARCH_SPACE["num_filters"], 30)
du_vals = np.linspace(*SEARCH_SPACE["dense_units"],  30)
NF, DU  = np.meshgrid(nf_vals, du_vals)

grid_4d = np.column_stack([
    np.full(NF.size, opt_bs),
    NF.ravel(),
    DU.ravel(),
    np.full(NF.size, opt_dr),
])
grid_transformed = opt_EI.space.transform(grid_4d.tolist())
mu, sigma = gp_model.predict(grid_transformed, return_std=True)
mu    = (-mu).reshape(NF.shape)
sigma = sigma.reshape(NF.shape)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

im0 = axes[0].contourf(NF, DU, mu, levels=20, cmap='viridis')
plt.colorbar(im0, ax=axes[0], label='Predicted Val Accuracy')
obs_nf = [xi[1] for xi in opt_EI.x_iters]
obs_du = [xi[2] for xi in opt_EI.x_iters]
obs_ac = [-y for y in opt_EI.func_vals]
sc0 = axes[0].scatter(obs_nf, obs_du, c=obs_ac, cmap='Reds',
                      edgecolors='white', s=70, zorder=5, label='BO EI observations')
plt.colorbar(sc0, ax=axes[0], label='Observed Val Accuracy')
axes[0].set_xlabel('num_filters')
axes[0].set_ylabel('dense_units')
axes[0].set_title(f'GP Surrogate Mean (EI)\n'
                  f'batch_size={opt_bs}, dropout={opt_dr:.2f} fixed at BO optimum')
axes[0].legend(loc='lower right')

im1 = axes[1].contourf(NF, DU, sigma, levels=20, cmap='plasma')
plt.colorbar(im1, ax=axes[1], label='Std. Deviation (uncertainty)')
axes[1].scatter(obs_nf, obs_du, color='white', edgecolors='black',
                s=70, zorder=5, label='BO EI observations')
axes[1].set_xlabel('num_filters')
axes[1].set_ylabel('dense_units')
axes[1].set_title('GP Surrogate Uncertainty (σ)\nHigh σ = unexplored regions')
axes[1].legend(loc='lower right')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'bo_surrogate_heatmap.png'), dpi=150)
plt.show()
print("Saved: bo_surrogate_heatmap.png")


# ── Plot 4: Hyperparameter exploration trajectories ───────────────────────────
param_names  = ['batch_size', 'num_filters', 'dense_units', 'dropout_rate']
param_bounds = [SEARCH_SPACE[p] for p in param_names]
random_vals  = [rand_bs, rand_nf, rand_du, rand_dr]
bo_vals_EI   = [[xi[j] for xi in opt_EI.x_iters] for j in range(4)]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.ravel()
full_xs = np.arange(1, N_ITER + 1)

for j, ax in enumerate(axes):
    ax.scatter(full_xs, random_vals[j],
               color='crimson',   alpha=0.7, s=50, label='Random Search')
    ax.scatter(full_xs, bo_vals_EI[j],
               color='steelblue', alpha=0.7, s=50, label='BO EI')
    ax.axvline(x=N_INIT, color='grey', linestyle='--', alpha=0.5,
               label=f'End of shared init (n={N_INIT})')
    ax.axhline(y=opt_EI.x[j], color='gold', linestyle=':', lw=2,
               label=f'BO best = {opt_EI.x[j]:.2g}')
    lo, hi = param_bounds[j]
    ax.set_ylim(lo - 0.05*(hi-lo), hi + 0.05*(hi-lo))
    ax.set_xlim(0, N_ITER + 1)
    ax.set_xlabel('Iteration')
    ax.set_ylabel(param_names[j])
    ax.set_title(f'Search trajectory: {param_names[j]}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle('Hyperparameter Exploration: Random Search vs BO EI',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'bo_exploration.png'), dpi=150)
plt.show()
print("Saved: bo_exploration.png")


# ── Summary ───────────────────────────────────────────────────────────────────
print("\n── Summary ──────────────────────────────────────────────────────────")
print(f"  Fixed LR: {FIXED_LR}")
print(f"  Random Search best val : {max_val_random[-1]:.4f}  params={best_random_params}")
print(f"  BO EI     best val     : {max(curve_EI):.4f}  params={opt_EI.x}")
print(f"  BO PI     best val     : {max(curve_PI):.4f}  params={opt_PI.x}")
print(f"  BO LCB    best val     : {max(curve_LCB):.4f}  params={opt_LCB.x}")
print()
for name, acc in test_results.items():
    print(f"  {name:<16s} final test acc: {acc:.4f}")
