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


# ── 4. Bayesian Optimisation with skopt ──────────────────────────────────────
print("\n── Bayesian Optimisation (optimiserer VALIDATION accuracy) ────────────")

bo_iter = [1]

def objective_function(x):
    batch_size, learning_rate, dropout_rate, num_filters = x
    val_acc = build_and_evaluate(batch_size, learning_rate, dropout_rate, num_filters)

    print(f"  Iter {bo_iter[0]:>2d}/{N_ITER} | "
          f"batch_size={batch_size:>4d} | "
          f"learning_rate={learning_rate:.6f} | "
          f"dropout_rate={dropout_rate:.2f} | "
          f"num_filters={num_filters:>3d} | "
          f"val_acc={val_acc:.4f}")
    bo_iter[0] += 1
    return -val_acc

np.int = int

# Seed BO korrekt med første random punkt (alle 4 parametre)
x0 = [[int(random_batch_sizes[0]), float(random_learning_rates[0]), float(random_dropout_rates[0]), int(random_num_filters[0])]]
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
best_bs, best_lr, best_dr, best_nf = opt.x
best_bo_val = -opt.fun

print(f"\n── Best hyperparametre (målt på VALIDATION) ─────────────────────────")
print(f"  Random best params: batch={best_random_params[0]}, lr={best_random_params[1]:.6f}, dr={best_random_params[2]:.2f}, nf={best_random_params[3]}")
print(f"  BO best params    : batch={best_bs}, lr={best_lr:.6f}, dr={best_dr:.2f}, nf={best_nf}")
print(f"  BO best val_acc   : {best_bo_val:.4f}")

# FINAL: Test evalueres én gang per metode
random_test_acc = train_final_and_test(*best_random_params, epochs=10)
bo_test_acc     = train_final_and_test(best_bs, best_lr, best_dr, best_nf, epochs=10)

print(f"\n── FINAL unbiased TEST (evalueret én gang) ───────────────────────────")
print(f"  Random Search TEST acc: {random_test_acc:.4f}")
print(f"  Bayesian Opt TEST acc : {bo_test_acc:.4f}")

# Export to csv
# 1. Prepare accuracy history data
xs = np.arange(1, N_ITER + 1)
# Calculate the cumulative maximum for the BO results
y_bo = np.maximum.accumulate(-opt.func_vals).ravel()
# Combine: Iteration number, Random Search Best, Bayesian Opt Best
results_hist = np.column_stack((xs, max_val_per_iter_random, y_bo))

# 2. Export accuracy history to CSV
np.savetxt("accuracy_history.csv", results_hist, delimiter=",", 
           header="Iteration,Random_Search_Acc,BO_Acc", comments="")

# 3. Export the hyperparameter path (the "knobs" used in each step)
path_data = np.array(opt.x_iters)
np.savetxt("optimization_path.csv", path_data, delimiter=",", 
           header="Batch_Size,Learning_Rate,Dropout_Rate,Num_Filters", comments="")

print("Data successfully exported to accuracy_history.csv and optimization_path.csv.")