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

# Setting Random State for reproducibility
RANDOM_SEED = 32
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# ── 1. Load Data ─────────────────────────────────────────────────────────────
(X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()
X_train_full = X_train_full.reshape(-1, 28, 28, 1).astype('float32') / 255.0
y_train_full = to_categorical(y_train_full, 10)

# Reverting to 10% data subset for a more robust evaluation
X_train_sub, X_val_sub, y_train_sub, y_val_sub = train_test_split(
    X_train_full, y_train_full, train_size=0.1, random_state=RANDOM_SEED
)

# ── 2. Search Space ──────────────────────────────────────────────────────────
space = [
    Integer(8, 512, name="batch_size"),
    Real(1e-6, 1e-1, prior="log-uniform", name="learning_rate"),
    Real(0.0, 0.5, name="dropout_rate"),
    Integer(8, 256, name="num_filters"),
]

def build_and_evaluate(bs, lr, dr, nf):
    model = Sequential([
        Input(shape=(28, 28, 1)),
        Conv2D(int(nf), (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dropout(dr),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy'])
    early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    
    # Reverting to 10 epochs
    history = model.fit(X_train_sub, y_train_sub, validation_data=(X_val_sub, y_val_sub), 
                        batch_size=int(bs), epochs=10, callbacks=[early_stop], verbose=0)
    return max(history.history['val_accuracy'])

# ── 3. Random Search ─────────────────────────────────────────────────────────
N_ITER = 20 # Reverting to 20 iterations
rng = np.random.default_rng(RANDOM_SEED)
max_val_per_iter_random = []
current_best_random = 0.0

print("── Running Random Search Baseline ──")
for i in range(N_ITER):
    bs = rng.integers(8, 512)
    lr = 10**rng.uniform(-6, -1)
    dr = rng.uniform(0, 0.5)
    nf = rng.integers(8, 256)
    
    val_acc = build_and_evaluate(bs, lr, dr, nf)
    current_best_random = max(current_best_random, val_acc)
    max_val_per_iter_random.append(current_best_random)
    print(f"Iter {i+1}/{N_ITER} | Best Random Acc: {current_best_random:.4f}")

# ── 4. Bayesian Optimization ─────────────────────────────────────────────────
print("\n── Running Bayesian Optimization ──")
def objective(x):
    val_acc = build_and_evaluate(*x)
    return -val_acc

res = gp_minimize(objective, space, n_calls=N_ITER, random_state=RANDOM_SEED, acq_func='EI')
y_bo = np.maximum.accumulate(-res.func_vals)

# ── 5. Clean CSV Export ──────────────────────────────────────────────────────
xs = np.arange(1, N_ITER + 1)
results_hist = np.column_stack((xs, max_val_per_iter_random, y_bo))
np.savetxt("accuracy_history.csv", results_hist, delimiter=",", 
           header="Iteration,Random_Search_Acc,BO_Acc", comments="", fmt=['%d', '%.4f', '%.4f'])

path_data = np.array(res.x_iters)
np.savetxt("optimization_path.csv", path_data, delimiter=",", 
           header="Batch_Size,Learning_Rate,Dropout_Rate,Num_Filters", comments="", 
           fmt=['%d', '%.6f', '%.4f', '%d'])

print("\nData exported to accuracy_history.csv and optimization_path.csv")