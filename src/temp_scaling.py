import numpy as np
import tensorflow as tf
import os
from scipy.optimize import minimize_scalar, minimize
from src.metrics import compute_aurc, compute_ece
from src.utils import softmax

def temp_scaling_nll(scores, true_y, iters=300):
    scores = tf.convert_to_tensor(scores, dtype=tf.float32, name='scores')
    true_y = tf.convert_to_tensor(tf.keras.utils.to_categorical(true_y), dtype=tf.float32)
    
    temp = tf.Variable(initial_value=1, trainable=True, dtype=tf.float32)

    def compute_temp_loss():
        scaled_logits = tf.math.divide(scores, temp)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=true_y, logits=scaled_logits))
        return loss

    optimizer = tf.optimizers.Adam(learning_rate=0.01)
    for _ in range(iters):
        optimizer.minimize(compute_temp_loss, var_list=[temp])

    return temp.numpy()

def temp_scaling_aurc(scores, pred_y, true_y):
    scores = np.array(scores)
    pred_y = np.array(pred_y)
    true_y = np.array(true_y)

    temps = np.linspace(0.01, 5.0, 100)
    best_temp = 1.0
    best_aurc = float('inf')

    for t in temps:
        val_class = softmax(scores / t)
        val = val_class[np.arange(len(pred_y)), pred_y]
        aurc = compute_aurc(val, pred_y, true_y)
        if aurc < best_aurc:
            best_aurc = aurc
            best_temp = t
    return best_temp

def temp_scaling_ece(scores, pred_y, true_y, conf_bin_num=10):
    temps = np.linspace(0.01, 5.0, 100)
    best_temp = 1.0
    best_ece = float('inf')

    for t in temps:
        val_class = softmax(scores / t)
        val = val_class[np.arange(len(pred_y)), pred_y]
        ece = compute_ece(val, pred_y, true_y, conf_bin_num)
        if ece < best_ece:
            best_ece = ece
            best_temp = t
    return best_temp

def ensemble_temp_scaling_nll(scores, true_y, num_classes, iters_temp=300, iters_weight=300):
    # Step 0: Clip logits to avoid overflow
    scores = tf.clip_by_value(scores, -50.0, 50.0)
    scores = tf.convert_to_tensor(scores, dtype=tf.float32)
    true_y = tf.convert_to_tensor(tf.keras.utils.to_categorical(true_y, num_classes), dtype=tf.float32)

    temp = tf.Variable(initial_value=1.0, trainable=True, dtype=tf.float32)

    def compute_temp_loss():
        scaled_logits = scores / temp
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=true_y, logits=scaled_logits))
        tf.debugging.check_numerics(loss, "Temp loss has NaNs or Infs")
        return loss

    optimizer = tf.optimizers.Adam(learning_rate=0.01)
    for _ in range(iters_temp):
        optimizer.minimize(compute_temp_loss, var_list=[temp])

    temp_val = temp.numpy()

    logits_np = scores.numpy()
    p1 = tf.nn.softmax(scores, axis=1).numpy()
    p0 = tf.nn.softmax(scores / temp_val, axis=1).numpy()
    p2 = np.ones_like(p0) / num_classes

    w = tf.Variable([1.0, 0.0, 0.0], dtype=tf.float32, trainable=True)

    def compute_weight_loss():
        p = w[0] * p0 + w[1] * p1 + w[2] * p2
        p = p / tf.reduce_sum(p, axis=1, keepdims=True)
        p = tf.clip_by_value(p, 1e-12, 1.0)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=true_y, logits=tf.math.log(p)))
        loss += 100.0 * tf.square(tf.reduce_sum(w) - 1.0)  # constraint: sum(w) == 1
        tf.debugging.check_numerics(loss, "Weight loss has NaNs or Infs")
        return loss

    optimizer_w = tf.optimizers.Adam(learning_rate=0.01)
    for _ in range(iters_weight):
        optimizer_w.minimize(compute_weight_loss, var_list=[w])

    w_val = w.numpy()

    if np.any(np.isnan(w_val)) or np.sum(w_val) == 0:
        print("Warning: NaNs in weights. Falling back to [1.0, 0.0, 0.0]")
        w_val = np.array([1.0, 0.0, 0.0])
    else:
        w_val = w_val / np.sum(w_val + 1e-8)

    return float(temp_val), w_val

class PTSCalibrator:
    """Parameterized Temperature Scaling (PTS) Calibrator"""

    def __init__(
        self,
        epochs=50,
        lr=1e-3,
        weight_decay=1e-4,
        batch_size=64,
        nlayers=2,
        n_nodes=16,
        length_logits=10,
        top_k_logits=5
    ):
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.nlayers = nlayers
        self.n_nodes = n_nodes
        self.length_logits = length_logits
        self.top_k_logits = top_k_logits

        self.model = self._build_model()

    def _build_model(self):
        input_logits = tf.keras.Input(shape=(self.length_logits,))
        l2_reg = tf.keras.regularizers.l2(self.weight_decay)

        # Extract top-k logits
        sorted_logits = tf.sort(input_logits, direction='DESCENDING')
        top_k = tf.keras.layers.Lambda(lambda x: x[:, :self.top_k_logits])(sorted_logits)

        x = top_k
        for _ in range(self.nlayers):
            x = tf.keras.layers.Dense(self.n_nodes, activation='relu', kernel_regularizer=l2_reg)(x)

        temperature = tf.keras.layers.Dense(1, activation=None, name="temperature")(x)
        temperature = tf.keras.layers.Lambda(lambda x: tf.math.abs(x))(temperature)

        # Normalize logits by temperature and apply softmax
        def scale_logits(inputs):
            logits, temp = inputs
            temp = tf.clip_by_value(temp, 1e-12, 1e12)
            return logits / temp

        scaled_logits = tf.keras.layers.Lambda(scale_logits)([input_logits, temperature])
        output_probs = tf.keras.layers.Softmax()(scaled_logits)

        model = tf.keras.Model(inputs=input_logits, outputs=output_probs)
        model.compile(optimizer=tf.keras.optimizers.Adam(self.lr),
                      loss=tf.keras.losses.MeanSquaredError())
        return model

    def tune(self, logits, labels, clip=1e2):
        if not tf.is_tensor(logits):
            logits = tf.convert_to_tensor(logits, dtype=tf.float32)

        logits = tf.clip_by_value(logits, -clip, clip)

        # Infer num_classes from logits
        num_classes = self.length_logits

        # Auto one-hot encode if needed
        if len(labels.shape) == 1:
            labels = tf.keras.utils.to_categorical(labels, num_classes=num_classes)

        if not tf.is_tensor(labels):
            labels = tf.convert_to_tensor(labels, dtype=tf.float32)

        assert logits.shape[1] == self.length_logits, "Logits must match specified input length"
        assert labels.shape[1] == self.length_logits, "Labels must match specified input length"

        self.model.fit(logits, labels, epochs=self.epochs, batch_size=self.batch_size, verbose=1)

    def calibrate(self, logits, clip=1e2):
        if not tf.is_tensor(logits):
            logits = tf.convert_to_tensor(logits, dtype=tf.float32)

        assert logits.shape[1] == self.length_logits, "Logits must match specified input length"

        logits = tf.clip_by_value(logits, -clip, clip)
        return self.model.predict(logits)

    def save(self, path="./"):
        if not os.path.exists(path):
            os.makedirs(path)
        filepath = os.path.join(path, "pts_model.h5")
        self.model.save_weights(filepath)
        print(f"Saved PTS model weights to: {filepath}")

    def load(self, path="./"):
        filepath = os.path.join(path, "pts_model.h5")
        self.model.load_weights(filepath)
        print(f"Loaded PTS model weights from: {filepath}")
