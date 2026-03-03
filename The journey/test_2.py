import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import skopt
from skopt import gp_minimize
from skopt.space import Integer, Real
from skopt.plots import plot_convergence
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

np.random.seed(32)
np.int = int  # numpy deprecation workaround required by skopt


# ── 1. Load & preprocess MNIST ───────────────────────────────────────────────
(train_X, train_y), (test_X, test_y) = mnist.load_data()

train_X = train_X.reshape(-1, 28, 28, 1).astype('float32') / 255.0
test_X  = test_X.reshape(-1,  28, 28, 1).astype('float32') / 255.0

train_y_cat = to_categorical(train_y, 10)
test_y_cat  = to_categorical(test_y,  10)

# ── Deliberately constrained subset to make the optimisation problem harder ──
# With 10 000 samples even bad architectures reach ~96 %, giving BO nothing
# to work with.  4 000 training samples + a shallow net spreads accuracy
# from ~82 % (bad config) to ~97 % (good config), so hyperparameter choices
# actually matter.
N_TRAIN, N_TEST = 4_000, 2_000
X_tr, y_tr = train_X[:N_TRAIN], train_y_cat[:N_TRAIN]
X_te, y_te = test_X[:N_TEST],   test_y_cat[:N_TEST]


# ── 2. CNN builder ────────────────────────────────────────────────────────────
# The four hyperparameters we optimise are all *architectural*:
#
#   batch_size   – how many samples per gradient update
#   num_filters  – conv filters; range 4–32 so small values genuinely hurt
#   dense_units  – width of the FC head; range 16–128
#   dropout_rate – regularisation strength; range 0.2–0.7 (high end hurts)
#
# Architecture is intentionally kept SHALLOW (single conv block) so that
# num_filters is the main capacity bottleneck and its effect is clearly visible
# in the surrogate heatmap.
#
# Learning rate is fixed at 1e-3.  It interacts with batch_size via the
# "linear scaling rule" and would dominate the BO landscape for the wrong
# reasons, obscuring the architectural story we want to tell.

FIXED_LR = 1e-3

def build_and_evaluate(
    batch_size:   int,
    num_filters:  int,
    dense_units:  int,
    dropout_rate: float,
    epochs:       int = 3,   # fewer epochs → under-configured nets don't recover
) -> float:
    """Build, train and evaluate a shallow CNN. Returns test accuracy."""
    model = Sequential([
        Input(shape=(28, 28, 1)),
        # Single conv block — shallow on purpose so num_filters is the bottleneck
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

    early_stop = EarlyStopping(monitor='val_accuracy', patience=2,
                               restore_best_weights=True)

    model.fit(
        X_tr, y_tr,
        batch_size=int(batch_size),
        epochs=epochs,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=0,
    )

    _, acc = model.evaluate(X_te, y_te, verbose=0)
    return float(acc)


# ── 3. Random Search baseline (20 iterations) ────────────────────────────────
print("\n── Random Search ──────────────────────────────────────────────────────")

N_ITER = 20
rng = np.random.default_rng(32)

random_params = {
    'batch_size':   rng.integers(16,  129, size=N_ITER),   # capped at 128; big batches hurt on small data
    'num_filters':  rng.integers(4,   33,  size=N_ITER),   # 4 filters is genuinely weak
    'dense_units':  rng.integers(16,  129, size=N_ITER),   # 16 units is very small
    'dropout_rate': rng.uniform(0.2,  0.7, size=N_ITER),   # 0.7 kills too much signal
}

current_best        = 0.0
max_acc_random      = []
random_all_accs     = []

for i in range(N_ITER):
    bs = int(random_params['batch_size'][i])
    nf = int(random_params['num_filters'][i])
    du = int(random_params['dense_units'][i])
    dr = float(random_params['dropout_rate'][i])

    acc = build_and_evaluate(bs, nf, du, dr)
    random_all_accs.append(acc)

    if acc > current_best:
        current_best = acc
    max_acc_random.append(current_best)

    print(f"  Iter {i+1:>2d}/{N_ITER} | "
          f"batch={bs:>4d} | filters={nf:>3d} | "
          f"dense={du:>4d} | dropout={dr:.2f} | "
          f"acc={acc:.4f} | best={current_best:.4f}")


# ── 4. Bayesian Optimisation with skopt ──────────────────────────────────────
print("\n── Bayesian Optimisation ──────────────────────────────────────────────")

space = [
    Integer(16,  128, name='batch_size'),
    Integer(4,   32,  name='num_filters'),
    Integer(16,  128, name='dense_units'),
    Real(0.2,    0.7, name='dropout_rate'),
]

bo_iter      = [1]
bo_all_accs  = []

def objective(x):
    batch_size, num_filters, dense_units, dropout_rate = x
    acc = build_and_evaluate(batch_size, num_filters, dense_units, dropout_rate)
    bo_all_accs.append(acc)

    print(f"  Iter {bo_iter[0]:>2d}/{N_ITER} | "
          f"batch={batch_size:>4d} | filters={num_filters:>3d} | "
          f"dense={dense_units:>4d} | dropout={dropout_rate:.2f} | "
          f"acc={acc:.4f}")
    bo_iter[0] += 1
    return -acc  # skopt minimises, so we negate accuracy

opt = gp_minimize(
    objective,
    space,
    n_calls=N_ITER,
    acq_func="EI",
    n_initial_points=5,   # first 5 are random; remaining 15 are guided by the GP
    random_state=32,
    xi=0.1,               # exploration–exploitation trade-off
    noise=0.01**2,
)

max_acc_bo = np.maximum.accumulate(-opt.func_vals).ravel()


# ── 5. Print best results ─────────────────────────────────────────────────────
print(f"\n── Best Results ─────────────────────────────────────────────────────")
print(f"  Random Search  – best acc: {max_acc_random[-1]:.4f}")
print(f"  Bayesian Opt.  – best acc: {max_acc_bo[-1]:.4f}")
print(f"  BO best params: batch={opt.x[0]}, filters={opt.x[1]}, "
      f"dense={opt.x[2]}, dropout={opt.x[3]:.2f}")


# ── 6. Plot 1: Convergence – Random Search vs BO ─────────────────────────────
xs = np.arange(1, N_ITER + 1)

plt.figure(figsize=(8, 5))
plt.plot(xs, max_acc_random, 'o-', color='crimson',    label='Random Search',          lw=2)
plt.plot(xs, max_acc_bo,     'o-', color='steelblue',  label='Bayesian Optimisation',  lw=2)
plt.axvline(x=5, color='steelblue', linestyle='--', alpha=0.5,
            label='BO: end of random init (n=5)')
plt.xlabel('Iterations')
plt.ylabel('Best Test Accuracy (cumulative)')
plt.title('Convergence: Random Search vs Bayesian Optimisation\n'
          '4 hyperparameters: batch_size, num_filters, dense_units, dropout_rate')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('bo_vs_random_convergence.png', dpi=150)
plt.show()
print("Saved: bo_vs_random_convergence.png")


# ── 7. Plot 2: Accuracy scatter per iteration ─────────────────────────────────
# Shows raw (non-cumulative) accuracy at every evaluation.
# BO should cluster near good regions faster than random search.

plt.figure(figsize=(8, 5))
plt.scatter(range(1, N_ITER+1), random_all_accs,
            color='crimson',   alpha=0.7, s=60, label='Random Search')
plt.scatter(range(1, N_ITER+1), bo_all_accs,
            color='steelblue', alpha=0.7, s=60, label='Bayesian Optimisation')
plt.axvline(x=5, color='steelblue', linestyle='--', alpha=0.5,
            label='BO: end of random init (n=5)')
plt.xlabel('Iteration')
plt.ylabel('Test Accuracy (this evaluation)')
plt.title('Per-Iteration Accuracy: Random Search vs Bayesian Optimisation')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('bo_vs_random_scatter.png', dpi=150)
plt.show()
print("Saved: bo_vs_random_scatter.png")


# ── 8. Plot 3: 2D GP surrogate heatmap ───────────────────────────────────────
# We slice the 4D surrogate across the two most expressive architectural axes
# (num_filters × dense_units) while fixing batch_size and dropout_rate at
# their optimal values found by BO.

gp_model = opt.models[-1]

# Optimal values to fix the other two dimensions
opt_bs = opt.x[0]
opt_dr = opt.x[3]

# Build a 2D grid over (num_filters, dense_units)
nf_vals = np.linspace(4,  32,  30)
du_vals = np.linspace(16, 128, 30)
NF, DU  = np.meshgrid(nf_vals, du_vals)

# Transform each dimension through skopt's internal scaler
grid_4d = np.column_stack([
    np.full(NF.size, opt_bs),
    NF.ravel(),
    DU.ravel(),
    np.full(NF.size, opt_dr),
])
grid_transformed = opt.space.transform(grid_4d.tolist())

mu, sigma = gp_model.predict(grid_transformed, return_std=True)
mu    = (-mu).reshape(NF.shape)      # flip sign back to accuracy
sigma = sigma.reshape(NF.shape)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Surrogate mean
im0 = axes[0].contourf(NF, DU, mu, levels=20, cmap='viridis')
plt.colorbar(im0, ax=axes[0], label='Predicted Accuracy')
axes[0].set_xlabel('num_filters')
axes[0].set_ylabel('dense_units')
axes[0].set_title(f'GP Surrogate Mean\n'
                  f'(batch_size={opt_bs}, dropout={opt_dr:.2f} fixed at BO optimum)')

# Overlay actual BO observations (projected onto this 2D slice)
obs_nf = [xi[1] for xi in opt.x_iters]
obs_du = [xi[2] for xi in opt.x_iters]
obs_ac = [-y for y in opt.func_vals]
sc0 = axes[0].scatter(obs_nf, obs_du, c=obs_ac, cmap='Reds',
                      edgecolors='white', s=70, zorder=5, label='BO observations')
plt.colorbar(sc0, ax=axes[0], label='Observed Accuracy')
axes[0].legend(loc='lower right')

# Surrogate uncertainty (std dev)
im1 = axes[1].contourf(NF, DU, sigma, levels=20, cmap='plasma')
plt.colorbar(im1, ax=axes[1], label='Std. Deviation (uncertainty)')
axes[1].set_xlabel('num_filters')
axes[1].set_ylabel('dense_units')
axes[1].set_title('GP Surrogate Uncertainty (σ)\n'
                  'High σ = unexplored regions the GP is uncertain about')
axes[1].scatter(obs_nf, obs_du, color='white', edgecolors='black',
                s=70, zorder=5, label='BO observations')
axes[1].legend(loc='lower right')

plt.tight_layout()
plt.savefig('bo_surrogate_heatmap.png', dpi=150)
plt.show()
print("Saved: bo_surrogate_heatmap.png")


# ── 9. Plot 4: Hyperparameter exploration over iterations ─────────────────────
# Shows which regions of each hyperparameter space BO explored vs random search.

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.ravel()

param_names  = ['batch_size', 'num_filters', 'dense_units', 'dropout_rate']
param_bounds = [(16, 128), (4, 32), (16, 128), (0.2, 0.7)]
random_vals  = [random_params['batch_size'],  random_params['num_filters'],
                random_params['dense_units'],  random_params['dropout_rate']]
bo_vals      = [[xi[j] for xi in opt.x_iters] for j in range(4)]

for j, ax in enumerate(axes):
    ax.scatter(xs, random_vals[j], color='crimson',   alpha=0.7, s=50, label='Random Search')
    ax.scatter(xs, bo_vals[j],     color='steelblue', alpha=0.7, s=50, label='Bayesian Opt.')
    ax.axvline(x=5, color='steelblue', linestyle='--', alpha=0.4)
    ax.axhline(y=opt.x[j], color='gold', linestyle=':', lw=2, label=f'BO best = {opt.x[j]:.2g}')
    ax.set_xlim(0, N_ITER + 1)
    ax.set_ylim(param_bounds[j][0] - 0.05*(param_bounds[j][1]-param_bounds[j][0]),
                param_bounds[j][1] + 0.05*(param_bounds[j][1]-param_bounds[j][0]))
    ax.set_xlabel('Iteration')
    ax.set_ylabel(param_names[j])
    ax.set_title(f'Search trajectory: {param_names[j]}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle('Hyperparameter Exploration: Random Search vs Bayesian Optimisation',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('bo_exploration.png', dpi=150)
plt.show()
print("Saved: bo_exploration.png")