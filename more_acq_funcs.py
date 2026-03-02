import numpy as np
import matplotlib.pyplot as plt
from skopt import gp_minimize
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from skopt.space import Integer, Real
from sklearn.model_selection import train_test_split
import tensorflow as tf

np.random.seed(32)
tf.random.set_seed(32)

# ─────────────────────────────────────────────────────────────────────────────
# SEARCH SPACE (ONE PLACE)
# ─────────────────────────────────────────────────────────────────────────────
SEARCH_SPACE = {
    "batch_size":   (8, 512),        # integer range
    "learning_rate": (1e-6, 1e-1),    # log-uniform range (min, max)  
    "dropout_rate": (0.0, 0.5),      # uniform range (min, max)
    "num_filters":  (8, 256),         # integer range for filters
}

# skopt space (bruges til BO)
space = [
    Integer(*SEARCH_SPACE["batch_size"], name="batch_size"),
    Real(*SEARCH_SPACE["learning_rate"], prior="log-uniform", name="learning_rate"),
    Real(*SEARCH_SPACE["dropout_rate"], name="dropout_rate"),
    Integer(*SEARCH_SPACE["num_filters"], name="num_filters"), 
]

# Random sampler (bruges til Random Search + til BO seed)
def sample_random_params(rng: np.random.Generator, n: int):
    bs_low, bs_high = SEARCH_SPACE["batch_size"]
    lr_low, lr_high = SEARCH_SPACE["learning_rate"]
    dr_low, dr_high = SEARCH_SPACE["dropout_rate"]
    nf_low, nf_high = SEARCH_SPACE["num_filters"]

    batch_sizes = rng.integers(bs_low, bs_high + 1, size=n)
    learning_rates = 10 ** rng.uniform(np.log10(lr_low), np.log10(lr_high), size=n)  # log-uniform
    dropout_rates = rng.uniform(dr_low, dr_high, size=n)
    num_filters = rng.integers(nf_low, nf_high + 1, size=n)  

    return batch_sizes, learning_rates, dropout_rates, num_filters


# ── 1. Load & preprocess MNIST ───────────────────────────────────────────────
(train_X, train_y), (test_X, test_y) = mnist.load_data()

print("Original MNIST Dataset:")
print('X_train:', train_X.shape)
print('Y_train:', train_y.shape)
print('X_test: ', test_X.shape)
print('Y_test: ', test_y.shape)

train_X = train_X.reshape(-1, 28, 28, 1).astype('float32') / 255.0
test_X  = test_X.reshape(-1,  28, 28, 1).astype('float32') / 255.0

train_y_cat = to_categorical(train_y, 10)
test_y_cat  = to_categorical(test_y,  10)

N_TRAIN = 1_000
N_TEST  = 200
X_tr, y_tr = train_X[:N_TRAIN], train_y_cat[:N_TRAIN]
X_te, y_te = test_X[:N_TEST],   test_y_cat[:N_TEST]

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
def build_model(learning_rate: float, dropout_rate: float, num_filters: int) -> Sequential:
    model = Sequential([
        Input(shape=(28, 28, 1)),
        Conv2D(int(num_filters), (3, 3), activation='relu'), 
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(dropout_rate),
        Dense(10, activation='softmax'),
    ])

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def build_and_evaluate(batch_size: int, learning_rate: float, dropout_rate: float, num_filters: int, epochs: int = 5) -> float:
    """Train på X_train og returnér VALIDATION accuracy (X_val)."""
    model = build_model(learning_rate=learning_rate, dropout_rate=dropout_rate, num_filters=num_filters)

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


def train_final_and_test(batch_size: int, learning_rate: float, dropout_rate: float, num_filters: int, epochs: int = 10) -> float:
    """Træn én gang på (train+val) og evaluer én gang på TEST."""
    X_final = np.concatenate([X_train, X_val], axis=0)
    y_final = np.concatenate([y_train, y_val], axis=0)

    model = build_model(learning_rate=learning_rate, dropout_rate=dropout_rate, num_filters=num_filters)

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

random_batch_sizes, random_learning_rates, random_dropout_rates, random_num_filters = sample_random_params(rng, N_ITER)

current_best_val = 0.0
best_random_params = None
max_val_per_iter_random = []

for i in range(N_ITER):
    bs = int(random_batch_sizes[i])
    lr = float(random_learning_rates[i])
    dr = float(random_dropout_rates[i])
    nf = int(random_num_filters[i])

    val_acc = build_and_evaluate(bs, lr, dr, nf)

    if val_acc > current_best_val:
        current_best_val = val_acc
        best_random_params = (bs, lr, dr, nf)

    max_val_per_iter_random.append(current_best_val)

    print(f"  Iter {i+1:>2d}/{N_ITER} | "
          f"batch_size={bs:>4d} | "
          f"learning_rate={lr:.6f} | "
          f"dropout_rate={dr:.2f} | "
          f"num_filters={nf:>3d} | "
          f"val_acc={val_acc:.4f} | "
          f"best_val={current_best_val:.4f}")


# ── 4. Bayesian Optimisation: compare EI vs PI vs LCB ─────────────────────────
print("\n── Bayesian Optimisation: compare EI vs PI vs LCB ─────────────────────")

from skopt import gp_minimize

# Keep things fair: same initial design points for all acquisition functions
N_INIT = 5                       # number of shared initial points
N_CALLS_TOTAL = N_ITER - N_INIT  # total evaluations per acquisition function

# Build shared x0 using your random sampler (first N_INIT points)
rng = np.random.default_rng(32)
rb, rl, rd, rnf = sample_random_params(rng, N_INIT)
x0 = [[int(rb[i]), float(rl[i]), float(rd[i]), int(rnf[i])] for i in range(N_INIT)]

# IMPORTANT: compute y0 ONCE and reuse for all (fair + faster overall)
# objective returns negative val_acc (because gp_minimize minimizes)
print(f"Computing shared y0 for {N_INIT} initial points...")
y0 = []
for i, x in enumerate(x0, 1):
    val_acc = build_and_evaluate(*x)
    y0.append(-val_acc)
    print(f"  init {i}/{N_INIT} | x={x} | val_acc={val_acc:.4f}")

# Helper to run BO with a given acquisition function
def run_bo(acq_func: str, xi=None, kappa=None):
    # gp_minimize will call our objective; keep per-run iteration printing clean
    it = {"i": 1}

    def objective(x):
        val_acc = build_and_evaluate(*x)
        print(f"  [{acq_func}] iter {it['i']:>2d}/{N_CALLS_TOTAL} | "
              f"bs={x[0]:>4d} lr={x[1]:.6f} dr={x[2]:.2f} nf={x[3]:>3d} | "
              f"val_acc={val_acc:.4f}")
        it["i"] += 1
        return -val_acc

    kwargs = dict(
        func=objective,
        dimensions=space,
        n_calls=N_CALLS_TOTAL,          # includes x0 points
        x0=x0,
        y0=y0,
        acq_func=acq_func,
        random_state=32,
        noise=0.01**2,
        # since we supply x0/y0, we don't need extra random init points:
        n_initial_points=0,
    )

    # Only pass the relevant parameter
    if acq_func in ("EI", "PI") and xi is not None:
        kwargs["xi"] = xi
    if acq_func == "LCB" and kappa is not None:
        kwargs["kappa"] = kappa

    return gp_minimize(**kwargs)

# Choose acquisition params (feel free to tweak)
xi_value = 0.1       # EI/PI exploration
kappa_value = 2.0    # LCB exploration

opt_EI  = run_bo("EI",  xi=xi_value)
opt_PI  = run_bo("PI",  xi=xi_value)
opt_LCB = run_bo("LCB", kappa=kappa_value)
# Extract "best val so far" curves (robust)
def best_curve(opt):
    vals = -np.array(opt.func_vals)          # convert to val_acc
    return np.maximum.accumulate(vals)

curves = {
    "EI":  best_curve(opt_EI),
    "PI":  best_curve(opt_PI),
    "LCB": best_curve(opt_LCB),
}

# Plot using per-curve x-axis lengths (no dimension mismatch)
plt.figure(figsize=(9, 5))
for name, curve in curves.items():
    xs = np.arange(1, len(curve) + 1)
    if name in ("EI", "PI"):
        label = f"BO {name} (xi={xi_value})"
    else:
        label = f"BO {name} (kappa={kappa_value})"
    plt.plot(xs, curve, 'o-', label=label)

plt.xlabel("Evaluations")
plt.ylabel("Best Validation Accuracy")
plt.title("Bayesian Optimisation: EI vs PI vs LCB (tuning on validation)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("bo_acq_compare.png", dpi=150)
plt.show()
print("Plot saved to bo_acq_compare.png")