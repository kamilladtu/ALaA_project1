import numpy as np
import matplotlib.pyplot as plt
import skopt
from skopt import gp_minimize
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from skopt.space import Integer, Real
from keras.layers import Input

np.random.seed(32)

# ── 1. Load & preprocess MNIST ───────────────────────────────────────────────
(train_X, train_y), (test_X, test_y) = mnist.load_data()

print('X_train:', train_X.shape)
print('Y_train:', train_y.shape)
print('X_test: ', test_X.shape)
print('Y_test: ', test_y.shape)

# Reshape to (N, 28, 28, 1) and normalise
train_X = train_X.reshape(-1, 28, 28, 1).astype('float32') / 255.0
test_X  = test_X.reshape(-1,  28, 28, 1).astype('float32') / 255.0

# One-hot encode labels - example the digit 3 becomes [0,0,0,1,0,0,0,0,0,0]
train_y_cat = to_categorical(train_y, 10)
test_y_cat  = to_categorical(test_y,  10)

# Use a subset for speed during BO (full dataset per eval is very slow)
N_TRAIN = 10_000
N_TEST  = 2_000
X_tr, y_tr = train_X[:N_TRAIN], train_y_cat[:N_TRAIN]
X_te, y_te = test_X[:N_TEST],   test_y_cat[:N_TEST]


# ── 2. CNN builder ────────────────────────────────────────────────────────────
def build_and_evaluate(batch_size: int, learning_rate: float, dropout_rate: float, epochs: int = 5) -> float:
    """Train a small CNN and return test accuracy."""
    model = Sequential([
    Input(shape=(28, 28, 1)),   # first layer
    Conv2D(32, (3, 3), activation='relu'), # reduces spacial size but keeps strong signals
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(dropout_rate), # randomly removes neurons during training to prevent overfitting
    Dense(10, activation='softmax'), # outputs class proberbilities
    ])

    # We want to optimize the learning rate when the CNN optimizes it's weights
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_accuracy', patience=2,
                               restore_best_weights=True)

    model.fit(X_tr, y_tr,
              batch_size=int(batch_size),
              epochs=epochs,
              validation_split=0.1,
              callbacks=[early_stop],
              verbose=0)

    _, acc = model.evaluate(X_te, y_te, verbose=0)
    return float(acc)


# ── 3. Random Search baseline (20 iterations) ────────────────────────────────
print("\n── Random Search ──────────────────────────────────────────────────────")

N_ITER = 20
rng = np.random.default_rng(32)
random_batch_sizes = rng.integers(16, 257, size=N_ITER)
random_learning_rates = 10 ** rng.uniform(-4, -2, size=N_ITER)
random_dropout_rates = rng.uniform(0.1, 0.5, size=N_ITER)

current_best_acc        = 0.0
max_acc_per_iter_random = []

for i in range(N_ITER):
    bs = int(random_batch_sizes[i])
    lr = float(random_learning_rates[i])
    dr = float(random_dropout_rates[i])

    acc = build_and_evaluate(bs, lr, dr)

    if acc > current_best_acc:
        current_best_acc = acc

    max_acc_per_iter_random.append(current_best_acc)

    print(f"  Iter {i+1:>2d}/{N_ITER} | "
      f"batch_size={bs:>4d} | "
      f"lr={lr:.5f} | "
      f"dropout={dr:.2f} | "
      f"acc={acc:.4f} | "
      f"best={current_best_acc:.4f}")


# ── 4. Bayesian Optimisation with skopt ──────────────────────────────────────
print("\n── Bayesian Optimisation ──────────────────────────────────────────────")

# Search spaces
space = [
    Integer(16, 256, name='batch_size'),
    Real(1e-4, 1e-2, prior='log-uniform', name='learning_rate'),
    Real(0.1, 0.5, name='dropout_rate')
]

# Seed BO with the first random search point so comparison is fair
x0 = [[random_batch_sizes[0]]]      # list of points, each point is a list
y0 = [-max_acc_per_iter_random[0]]  # flat scalar (skopt minimises, so negate accuracy)

bo_iter = [1]  # use list to allow mutation inside nested function

def objective_function(x):
    """Objective for skopt: takes [batch_size] and [learning_rate], returns -accuracy."""
    batch_size  = x[0]
    learning_rate = x[1]
    dropout_rate = x[2]

    acc = build_and_evaluate(batch_size, learning_rate, dropout_rate)
    print(f"  Iter {bo_iter[0]:>2d}/20 | "
          f"batch_size={batch_size:>4d} | "
          f"learning_rate={learning_rate:.5f} | "
          f"dropout_rate={dropout_rate:.2f} | "
          f"acc={acc:.4f}")
    bo_iter[0] += 1
    return -acc

np.int = int  # numpy deprecation workaround required by skopt

opt = gp_minimize(
    objective_function,
    space,
    n_calls=20,
    acq_func="EI",
    n_initial_points=5,
    random_state=32,
    xi=0.1,
    noise=0.01**2,
)


# ── 5. Results ────────────────────────────────────────────────────────────────
best_bs  = opt.x[0]
best_acc = -opt.fun

print(f"\n── Best result ──────────────────────────────────────────────────────")
print(f"  Best batch_size : {best_bs}")
print(f"  Best test acc   : {best_acc:.4f}")


# ── 6. Plot 1: Random Search vs Bayesian Optimisation ────────────────────────
y_bo = np.maximum.accumulate(-opt.func_vals).ravel()
xs   = np.arange(1, N_ITER + 1)

plt.figure(figsize=(8, 5))
plt.plot(xs, max_acc_per_iter_random, 'o-', color='red',  label='Random Search')
plt.plot(xs, y_bo,                   'o-', color='blue', label='Bayesian Optimisation')
plt.xlabel('Iterations')
plt.ylabel('Best Test Accuracy')
plt.title('Random Search vs Bayesian Optimisation')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('bo_vs_random.png', dpi=150)
plt.show()
print("Plot saved to bo_vs_random.png")


# ── 7. Plot 2: GP Surrogate + Acquisition Function ───────────────────────────
gp_model  = opt.models[-1]
grid      = np.arange(16, 257).reshape(-1, 1)
grid_norm = np.array([opt.space[0][1].transform(x) for x in grid.ravel()]).reshape(-1, 1)

ye, ye_std = gp_model.predict(X=grid_norm, return_std=True)
ye     = -np.array(ye).ravel()     # flip sign back to accuracy
ye_std =  np.array(ye_std).ravel()

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Surrogate mean
axes[0].plot(grid, ye, color='steelblue')
axes[0].fill_between(grid.ravel(), ye - 2*ye_std, ye + 2*ye_std,
                     alpha=0.2, color='steelblue', label='±2σ')
axes[0].scatter([x[0] for x in opt.x_iters], -opt.func_vals,
                color='red', zorder=5, label='Observations')
axes[0].set_xlabel('Batch Size')
axes[0].set_ylabel('Predicted Accuracy')
axes[0].set_title('Surrogate Function (GP mean)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Surrogate std
axes[1].plot(grid, ye_std, color='orange')
axes[1].set_xlabel('Batch Size')
axes[1].set_ylabel('Standard Deviation')
axes[1].set_title('Surrogate Function Std Dev')
axes[1].grid(True, alpha=0.3)

# Acquisition function (EI)
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