import tensorflow as tf
import numpy as np
from tqdm import tqdm
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam

#----------------------------
# Compute PMI
#----------------------------

def neural_pmi(x, y, pmi_model, estimator):
    scores = pmi_model(x,y)
    if estimator == 'probabilistic_classifier':
        batch_size = scores.shape[0]
        pmi = tf.linalg.diag_part(scores) + tf.math.log(tf.cast(batch_size - 1, dtype=tf.float32))
#         joint_logits = tf.sigmoid(tf.reshape(scores, [-1])[::(batch_size + 1)])
#         pmi = tf.math.log((batch_size-1.)*joint_logits/(1.-joint_logits))    # N_pxpy / N_pxy = (batch_size - 1.) * batch_size / batch_size
        # we have batch_size*(batch_size-1) product of marginal samples
        # we have batch_size joint density samples
    elif estimator == 'density_ratio_fitting':
        pmi = tf.linalg.diag_part(tf.math.log(tf.maximum(scores, 1e-4)))
    elif estimator == 'variational_f_js':
        pmi = tf.linalg.diag_part(scores)
    else:
        raise NotImplementedError(f"Estimator ({estimator}) not supported.")
    if scores.shape[0] == 1:
        pmi = scores.numpy()[0]
    return np.array(pmi)

#----------------------------
# Train PMI Estimator
#----------------------------

def train_critic_model(ds_trn, ds_val, critic, estimator, epochs, save_path):
    if critic == 'concat':
        model = ConcatCritic(ds_trn)
    elif critic == 'separable':
        model = SeparableCritic(ds_trn)
    else:
        raise NotImplementedError(f"Critic model ({critic}) not supported.")
    
    if estimator =='probabilistic_classifier':
        loss_fn = probabilistic_classifier_obj
    elif estimator == 'density_ratio_fitting':
        loss_fn = density_ratio_fitting_obj
    elif estimator =='variational_f_js':
        loss_fn = js_fgan_lower_bound_obj
    else:
        raise NotImplementedError(f"Estimator ({estimator}) not supported.")
        
    optimizer = Adam(learning_rate=0.001)
    
    @tf.function
    def train_step(x, y, model, optimizer, loss_fn):
        with tf.GradientTape() as tape:
            scores = model(x,y)
            loss_value = -loss_fn(scores)
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        return -loss_value
    
    if ds_val is not None:
        max_mi = 0
        patience = 10
        wait = 0 
        for epoch in tqdm(range(epochs), desc="Epochs"):
            for step, (x_batch, y_batch) in enumerate(ds_trn):
                negative_loss = train_step(x_batch, y_batch, model, optimizer, loss_fn)
            pmi_list = []
            for x_batch, y_batch in ds_val:
                pmi_list.extend(neural_pmi(x_batch, y_batch, model, estimator=estimator).tolist())
            current_mi = np.mean(pmi_list)
            if current_mi > max_mi:
                max_mi = current_mi
                model.save(save_path)
                wait = 0 
            else:
                wait += 1 
            if wait >= patience:
                break
    else:
        for epoch in range(epochs):
            for step, (x_batch, y_batch) in enumerate(ds_trn):
                negative_loss = train_step(x_batch, y_batch, model, optimizer, loss_fn)
        model.save(save_path)

#----------------------------
# Critic Architectures
#----------------------------

def mlp_critic(input_dim, output_dim):
    model = tf.keras.Sequential()
    model.add(Dense(512, activation='relu', input_shape=(input_dim,)))
    model.add(Dense(output_dim))
    return model

class SeparableCritic(Model):
    # pass x to g and pass y to h --> f(x,y) = g(x)^T h(y) --> only require 2N forward passes
    def __init__(self, dataset, output_dim=128, **extra_kwargs):
        super(SeparableCritic, self).__init__()
        dim_x = dataset.element_spec[0].shape[1]
        dim_y = dataset.element_spec[1].shape[1]
        self.output_dim = output_dim
        self._g = mlp_critic(dim_x, self.output_dim)
        self._h = mlp_critic(dim_y, self.output_dim)
    def call(self, x, y):
        g_output = self._g(x)
        h_output = self._h(y)
        scores = tf.matmul(h_output, tf.transpose(g_output))
        return scores   # shape = (batch_size, batch_size)
    def get_config(self):
        config = super(SeparableCritic, self).get_config()
        config.update({
            'output_dim': self.output_dim,
            'g': self._g.get_config(),
            'h': self._h.get_config()
        })
        return config

class ConcatCritic(Model):
    # concatenate x and y --> require batch_size^2 forward passes
    def __init__(self, dataset, **extra_kwargs):
        super(ConcatCritic, self).__init__()
        dim_x = dataset.element_spec[0].shape[1]
        dim_y = dataset.element_spec[1].shape[1]
        self._f = mlp_critic(dim_x+dim_y, 1)  # output is scalar score
    def call(self, x, y):
        # shape of x: (batch_size, dim_x)
        # shape of y: (batch_size, dim_y)
        batch_size = tf.shape(x)[0]
        x_tiled = tf.tile(tf.expand_dims(x, axis=1), [1, batch_size, 1])    # shape = (batch_size, batch_size, dim_x)
        y_tiled = tf.tile(tf.expand_dims(y, axis=0), [batch_size, 1, 1])    # shape = (batch_size, batch_size, dim_y)
        xy_pairs = tf.concat([x_tiled, y_tiled], axis=-1)                   # shape = (batch_size, batch_size, dim_x+dim_y)
        scores = self._f(tf.reshape(xy_pairs, [batch_size * batch_size, -1]))
        return tf.reshape(scores, [batch_size, batch_size])                 # shape = (batch_size, batch_size)
    def get_config(self):
        config = super(ConcatCritic, self).get_config()
        config['f'] = self._f
        return config

#----------------------------
# Training Objectives
#----------------------------

# Method 1: Probabilistic Classifier
@tf.function
def probabilistic_classifier_obj(score):
    criterion = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    batch_size = score.shape[0]
    labels = [0.]*(batch_size*batch_size)
    labels[::(batch_size+1)] = [1.]*batch_size      # assign label 0 to samples drawn from product of marginals and label 1 to samples drawn from joint density
    labels = tf.convert_to_tensor(labels, dtype=score.dtype)
    labels = tf.reshape(labels, (-1, 1))
    logits = tf.reshape(score, (-1, 1))
    loss = -1.*criterion(labels, logits)
    return loss

# Method 2: Density Ratio Fitting
@tf.function
def density_ratio_fitting_obj(score):
    score_square = tf.square(score)
    batch_size = score.shape[0]  # batch_size
    joint_term = tf.reduce_mean(tf.linalg.diag_part(score))
    marg_term = ((tf.reduce_sum(score_square) - tf.reduce_sum(tf.linalg.diag_part(score_square))) / (batch_size*(batch_size-1.)))
    return joint_term - 0.5*marg_term

# Method 3: Variational Representation of f-divergence (JS)
@tf.function
def js_fgan_lower_bound_obj(score):
    score_diag = tf.linalg.diag_part(score)
    first_term = -tf.reduce_mean(tf.nn.softplus(-score_diag))
    batch_size = score.shape[0]
    second_term = (tf.reduce_sum(tf.nn.softplus(score)) - tf.reduce_sum(tf.nn.softplus(score_diag))) / (batch_size * (batch_size - 1.))
    return first_term - second_term
