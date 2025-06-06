import tensorflow as tf
import numpy as np
from sklearn.isotonic import IsotonicRegression

def model_logits(model, ds):
    return model.predict(ds.batch(512), verbose=0)

def softmax_prob(model, ds, opt_temp=None):
    logits = model_logits(model, ds)
    if opt_temp is not None:
        logits = logits / opt_temp
    prob = tf.nn.softmax(logits, axis=-1)
    return prob

def max_softmax_prob(model, ds, opt_temp=None):
    logits = model_logits(model, ds)
    if opt_temp is not None:
        logits = logits / opt_temp
    max_prob = np.max(tf.nn.softmax(logits, axis=-1), axis=1)
    return max_prob
    
def softmax_margin(model, ds, opt_temp=None):
    logits = model_logits(model, ds)
    if opt_temp is not None:
        logits = logits / opt_temp
    softmax_probs = tf.nn.softmax(logits, axis=-1)
    top2 = tf.math.top_k(softmax_probs, k=2).values
    margins = tf.subtract(top2[:, 0], top2[:, 1])
    return margins.numpy()
    
def max_logits(model, ds):
    logits = model_logits(model, ds)
    return np.max(logits, axis=1)

def logits_margin(model, ds):
    logits = model_logits(model, ds)
    top2 = tf.math.top_k(logits, k=2).values
    margins = tf.subtract(top2[:, 0], top2[:, 1])
    return margins.numpy()

def negative_entropy(model, ds, opt_temp=None):
    logits = model_logits(model, ds)
    if opt_temp is not None:
        logits = logits / opt_temp
    softmax_probs = tf.nn.softmax(logits, axis=-1)
    neg_ent = tf.reduce_sum(softmax_probs * tf.math.log(softmax_probs + 1e-9), axis=-1)
    return neg_ent.numpy()

def negative_gini(model, ds, opt_temp=None):
    logits = model_logits(model, ds)
    if opt_temp is not None:
        logits = logits / opt_temp
    softmax_probs = tf.nn.softmax(logits, axis=-1)
    neg_gini = tf.reduce_sum(tf.square(softmax_probs), axis=-1) - 1
    return neg_gini.numpy()

def isotonic_reg(model, ds_val, ds_test, true_y_val):
    logits_val = model_logits(model, ds_val)
    max_prob_val = np.max(tf.nn.softmax(logits_val, axis=-1), axis=1)
    logits_test = model_logits(model, ds_test)
    max_prob_test = np.max(tf.nn.softmax(logits_test, axis=-1), axis=1)
    
    pred_y_val = np.argmax(logits_val, axis=1)
    correctness = (pred_y_val == true_y_val).astype(int)
    
    iso_reg = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1)
    iso_reg.fit(max_prob_val, correctness)

    calibrated_probs = iso_reg.predict(max_prob_test)

    return calibrated_probs
