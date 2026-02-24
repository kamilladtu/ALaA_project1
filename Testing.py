import numpy as np
import matplotlib.pyplot as plt
import skopt
from skopt import gp_minimize
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from skopt.space import Integer, Real
from sklearn.model_selection import train_test_split

np.random.seed(32)

# ─────────────────────────────────────────────────────────────────────────────
# SEARCH SPACE
# ─────────────────────────────────────────────────────────────────────────────
SEARCH_SPACE = {
    "batch_size": (16, 256),     # integer range
    "learning_rate": (1e-4, 100),  # log-uniform range (min, max)
    "dropout_rate": (0.01, 0.5)   # uniform range (min, max)
}

# skopt space (bruges til BO)
space = [
    Integer(SEARCH_SPACE["batch_size"][0], SEARCH_SPACE["batch_size"][1], name="batch_size"),
    Real(SEARCH_SPACE["learning_rate"][0], SEARCH_SPACE["learning_rate"][1], prior="log-uniform", name="learning_rate"),
    Real(SEARCH_SPACE["dropout_rate"][0], SEARCH_SPACE["dropout_rate"][1], name="dropout_rate"),
]

# Random sampler (bruges til Random Search + til BO seed)
def sample_random_params(rng: np.random.Generator, n: int):
    bs_low, bs_high = SEARCH_SPACE["batch_size"]
    lr_low, lr_high = SEARCH_SPACE["learning_rate"]
    dr_low, dr_high = SEARCH_SPACE["dropout_rate"]

    batch_sizes = rng.integers(bs_low, bs_high + 1, size=n)  # +1 fordi integers high er exclusive
    learning_rates = 10 ** rng.uniform(np.log10(lr_low), np.log10(lr_high), size=n)  # log-uniform
    dropout_rates = rng.uniform(dr_low, dr_high, size=n)

    return batch_sizes, learning_rates, dropout_rates


# ── 1. Load & preprocess MNIST ───────────────────────────────────────────────
(train_X, train_y), (test_X, test_y) = mnist.load_data()

print("Original MNIST Dataset:")
print('X_train:', train_X.shape)
print('Y_train:', train_y.shape)
print('X_test: ', test_X.shape)
print('Y_test: ', test_y.shape)

# Reshape to (N, 28, 28, 1) and normalise
train_X = train_X.reshape(-1, 28, 28, 1).astype('float32') / 255.0
test_X  = test_X.reshape(-1,  28, 28, 1).astype('float32') / 255.0

# One-hot encode labels
train_y_cat = to_categorical(train_y, 10)
test_y_cat  = to_categorical(test_y,  10)

# Use only a subset of original dataset for better speed during tuning
N_TRAIN = 10_000  # subset for training+tuning
N_TEST  = 2_000   # subset for final test evaluation
X_tr, y_tr = train_X[:N_TRAIN], train_y_cat[:N_TRAIN]
X_te, y_te = test_X[:N_TEST],   test_y_cat[:N_TEST]

# Fixed train/validation split ONCE
X_train, X_val, y_train, y_val = train_test_split(
    X_tr, y_tr,
    test_size=0.2,
    random_state=32,
    stratify=train_y[:N_TRAIN]
)

print("\nTuning split:")
print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_val:  ", X_val.shape,   "y_val:  ", y_val.shape)
print("X_test: ", X_te.shape,    "y_test: ", y_te.shape)


# ── 2. CNN builder ────────────────────────────────────────────────────────────
def build_model(learning_rate: float, dropout_rate: float) -> Sequential:
    model = Sequential([
        Input(shape=(28, 28, 1)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(dropout_rate),
        Dense(10, activation='softmax'),
    ])

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def build_and_evaluate(batch_size: int, learning_rate: float, dropout_rate: float, epochs: int = 5) -> float:
    """Train på X_train og returnér VALIDATION accuracy (X_val)."""
    model = build_model(learning_rate=learning_rate, dropout_rate=dropout_rate)

    early_stop = EarlyStopping(monitor='val_accuracy', patience=2, restore_best_weights=True)

    model.fit(
        X_train, y_train,
        batch_size=int(batch_size),
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=[early_stop],
        verbose=0
    )

    _, val_acc = model.evaluate(X_val, y_val, verbose=0)
    return float(val_acc)


def train_final_and_test(batch_size: int, learning_rate: float, dropout_rate: float, epochs: int = 10) -> float:
    """Træn én gang på (train+val) og evaluer én gang på TEST."""
    X_final = np.concatenate([X_train, X_val], axis=0)
    y_final = np.concatenate([y_train, y_val], axis=0)

    model = build_model(learning_rate=learning_rate, dropout_rate=dropout_rate)

    model.fit(
        X_final, y_final,
        batch_size=int(batch_size),
        epochs=epochs,
        verbose=0
    )

    _, test_acc = model.evaluate(X_te, y_te, verbose=0)
    return float(test_acc)


# ── 3. Random Search baseline ────────────────────────────────────────────────
print("\n── Random Search (optimiserer VALIDATION accuracy) ───────────────────")

N_ITER = 20
rng = np.random.default_rng(32)

random_batch_sizes, random_learning_rates, random_dropout_rates = sample_random_params(rng, N_ITER)

current_best_val = 0.0
best_random_params = None
max_val_per_iter_random = []

for i in range(N_ITER):
    bs = int(random_batch_sizes[i])
    lr = float(random_learning_rates[i])
    dr = float(random_dropout_rates[i])

    val_acc = build_and_evaluate(bs, lr, dr)

    if val_acc > current_best_val:
        current_best_val = val_acc
        best_random_params = (bs, lr, dr)

    max_val_per_iter_random.append(current_best_val)

    print(f"  Iter {i+1:>2d}/{N_ITER} | "
          f"batch_size={bs:>4d} | "
          f"learning_rate={lr:.6f} | "
          f"dropout_rate={dr:.2f} | "
          f"val_acc={val_acc:.4f} | "
          f"best_val={current_best_val:.4f}")


# ── 4. Bayesian Optimisation with skopt ──────────────────────────────────────
print("\n── Bayesian Optimisation (optimiserer VALIDATION accuracy) ────────────")

bo_iter = [1]

def objective_function(x):
    batch_size, learning_rate, dropout_rate = x
    val_acc = build_and_evaluate(batch_size, learning_rate, dropout_rate)

    print(f"  Iter {bo_iter[0]:>2d}/{N_ITER} | "
          f"batch_size={batch_size:>4d} | "
          f"learning_rate={learning_rate:.6f} | "
          f"dropout_rate={dropout_rate:.2f} | "
          f"val_acc={val_acc:.4f}")
    bo_iter[0] += 1
    return -val_acc  # skopt minimerer

np.int = int  # numpy deprecation workaround required by skopt

# ✅ Seed BO korrekt med første random punkt (alle 3 parametre)
x0 = [[int(random_batch_sizes[0]), float(random_learning_rates[0]), float(random_dropout_rates[0])]]
y0 = [-build_and_evaluate(*x0[0])]

opt = gp_minimize(
    objective_function,
    space,
    n_calls=N_ITER - 1,     # 1 seed + (N_ITER-1) = N_ITER total
    x0=x0,
    y0=y0,
    acq_func="EI",
    n_initial_points=5,
    random_state=32,
    xi=0.1,
    noise=0.01**2,
)

# ── 5. Resultater ────────────────────────────────────────────────────────────
best_bs, best_lr, best_dr = opt.x
best_bo_val = -opt.fun

print(f"\n── Best hyperparametre (målt på VALIDATION) ─────────────────────────")
print(f"  Random best params: batch={best_random_params[0]}, lr={best_random_params[1]:.6f}, dr={best_random_params[2]:.2f}")
print(f"  BO best params    : batch={best_bs}, lr={best_lr:.6f}, dr={best_dr:.2f}")
print(f"  BO best val_acc   : {best_bo_val:.4f}")

# ✅ FINAL: Test evalueres én gang per metode
random_test_acc = train_final_and_test(*best_random_params, epochs=10)
bo_test_acc     = train_final_and_test(best_bs, best_lr, best_dr, epochs=10)

print(f"\n── FINAL unbiased TEST (evalueret én gang) ───────────────────────────")
print(f"  Random Search TEST acc: {random_test_acc:.4f}")
print(f"  Bayesian Opt TEST acc : {bo_test_acc:.4f}")


# ── 6. Plot 1: Random Search vs Bayesian Optimisation (VAL curves) ───────────
y_bo = np.maximum.accumulate(-opt.func_vals).ravel()
xs = np.arange(1, N_ITER + 1)

plt.figure(figsize=(8, 5))
plt.plot(xs, max_val_per_iter_random, 'o-', color='red',  label='Random Search (best val)')
plt.plot(xs, y_bo,                    'o-', color='blue', label='Bayesian Optimisation (best val)')
plt.xlabel('Iterations')
plt.ylabel('Best Validation Accuracy')
plt.title('Random Search vs Bayesian Optimisation (tuning on validation)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('bo_vs_random.png', dpi=150)
plt.show()
print("Plot saved to bo_vs_random.png")


# ── 7. Plot 2: GP Surrogate + Acquisition Function (kun batch dimension) ─────
# NB: Denne plot giver kun mening hvis man “holder de andre parametre faste”.
# Her viser vi stadig kun batch-dimensionen, som du gjorde før.
gp_model  = opt.models[-1]

bs_low, bs_high = SEARCH_SPACE["batch_size"]
grid      = np.arange(bs_low, bs_high + 1).reshape(-1, 1)
grid_norm = np.array([opt.space[0][1].transform(x) for x in grid.ravel()]).reshape(-1, 1)

ye, ye_std = gp_model.predict(X=grid_norm, return_std=True)
ye     = -np.array(ye).ravel()
ye_std =  np.array(ye_std).ravel()

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

axes[0].plot(grid, ye, color='steelblue')
axes[0].fill_between(grid.ravel(), ye - 2*ye_std, ye + 2*ye_std,
                     alpha=0.2, color='steelblue', label='±2σ')
axes[0].scatter([x[0] for x in opt.x_iters], -opt.func_vals,
                color='red', zorder=5, label='Observations')
axes[0].set_xlabel('Batch Size')
axes[0].set_ylabel('Predicted Validation Accuracy')
axes[0].set_title('Surrogate Function (GP mean)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(grid, ye_std, color='orange')
axes[1].set_xlabel('Batch Size')
axes[1].set_ylabel('Standard Deviation')
axes[1].set_title('Surrogate Function Std Dev')
axes[1].grid(True, alpha=0.3)

acq_val = skopt.acquisition.gaussian_ei(
    X=grid_norm, model=gp_model, xi=0.1,
    y_opt=opt.fun, return_grad=False
)
axes[2].plot(grid, acq_val, color='green')
axes[2].set_xlabel('Batch Size')
axes[2].set_ylabel('Expected Improvement')
axes[2].set_title('Acquisition Function (EI)')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('bo_surrogate.png', dpi=150)
plt.show()
print("Plot saved to bo_surrogate.png")