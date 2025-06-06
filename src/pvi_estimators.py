import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout

from src.models import get_optimizer
from src.datasets import prefetch_dataset
from src.utils import softmax

#----------------------------
# Train PVI null model
#----------------------------

def train_pvi_null_model(dataset, model, epochs, save_path):
    ds_null = dataset.map(lambda x, y: (tf.zeros_like(x), y))
    ds_null = prefetch_dataset(ds_null, batch_size=128)
    optimizer = tf.keras.optimizers.Adam(0.001)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    model.compile(loss=loss_fn, optimizer=optimizer, metrics=['accuracy'])
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, min_delta=0.001, restore_best_weights=True)
    model.fit(ds_null, verbose=1, epochs=epochs, callbacks=[early_stop,])
    model.save_weights(save_path)
    
#----------------------------
# Train PVI MLP model
#----------------------------

def train_pvi_model(dataset, model, cfg, epochs, save_path):
    optimizer = tf.keras.optimizers.Adam(0.001)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    model.compile(loss=loss_fn, optimizer=optimizer, metrics=['accuracy'])
    model.fit(dataset, verbose=1, epochs=epochs)
    model.save(save_path)
    
#----------------------------
# Compute PVI for 1 class
#----------------------------

def v_entropy(x,y,model):
    prob = tf.nn.softmax(model.predict(x)).numpy()
    return -1 * np.log(tf.boolean_mask(prob, y).numpy())

def neural_pvi(x,y,model,null_model):
    null_x = np.zeros_like(x)
    v_cond_entropy = v_entropy(x,y,model)
    v_null_entropy = v_entropy(null_x,y,null_model)
    pvi = v_null_entropy - v_cond_entropy
    return pvi

#----------------------------
# Compute PVI for all classes
#----------------------------

def v_entropy_class(ds,model,temp=1,eps=1e-40):
    logits = model.predict(ds) / temp
    prob = tf.nn.softmax(logits).numpy()
    prob = np.clip(prob, eps, 1.0)
    return -1 * np.log(prob)

def neural_pvi_class(ds,model,null_model,opt_temp_pvi=1,opt_temp_null=1):
    ds_null = ds.map(lambda x, y: (tf.zeros_like(x), y))
    v_cond_entropy = v_entropy_class(ds,model,opt_temp_pvi)
    v_null_entropy = v_entropy_class(ds_null,null_model,opt_temp_null)
    pvi = v_null_entropy - v_cond_entropy
    return pvi

#------------------------------
# Compute PVI for saliency maps
#------------------------------

def train_pvi_saliency_null_model(images, labels, cfg, epochs, save_path):
    null_images = np.zeros_like(images)
    ds_null = tf.data.Dataset.from_tensor_slices((null_images, labels))
    ds_null = ds_null.batch(cfg['batch_size'])
    model = tf.keras.Sequential()
    model.add(Input(shape=(ds_null.element_spec[0].shape[1],)))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(ds_null.element_spec[1].shape[1], activation='softmax'))
    optimizer = tf.keras.optimizers.Adam(0.001)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(loss=loss_fn, optimizer=optimizer, metrics=['accuracy'])
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, min_delta=0.001, restore_best_weights=True)
    model.fit(ds_null, verbose=1, epochs=epochs, callbacks=[early_stop,])
    model.save(save_path)

def train_pvi_saliency_model(ds_train, cfg, epochs, save_path):
    model = tf.keras.Sequential()
    model.add(Input(shape=(ds_train.element_spec[0].shape[1],)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(ds_train.element_spec[1].shape[1], activation='softmax'))
    optimizer = tf.keras.optimizers.Adam(0.001)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(loss=loss_fn, optimizer=optimizer, metrics=['accuracy'])
    model.fit(ds_train, verbose=1, epochs=epochs)
    model.save(save_path)

#----------------------------
# Ensembling Approaches
#----------------------------

def v_entropy_ensemble(xs,y,models,eps=1e-40):
    preds = tf.convert_to_tensor([model.predict(x) for model, x in zip(models, xs)])
    probs = tf.nn.softmax(preds, axis=-1)
    avg_prob = tf.reduce_mean(probs, axis=0).numpy()
    avg_prob = np.clip(avg_prob, eps, 1.0)
    return -1 * np.log(tf.boolean_mask(avg_prob, y).numpy())

def neural_pvi_ensemble(xs,y,models,null_models):
    nulls = np.array([np.zeros_like(x) for x in xs])
    v_cond_entropy = v_entropy_ensemble(xs,y,models)
    v_null_entropy = v_entropy_ensemble(nulls,y,null_models)
    pvi = v_null_entropy - v_cond_entropy
    return pvi

def v_entropy_ensemble_class(xs, models, temps=None, eps=1e-40):
    preds = tf.convert_to_tensor([
        model.predict(x) / t if temps is not None else model.predict(x)
        for model, x, t in zip(models, xs, temps if temps is not None else [None]*len(models))
    ])
    probs = tf.nn.softmax(preds, axis=-1)
    avg_prob = tf.reduce_mean(probs, axis=0).numpy()
    avg_prob = np.clip(avg_prob, eps, 1.0)
    return -1 * np.log(avg_prob)

def neural_pvi_ensemble_class(xs, models, null_models, temps=None, null_temps=None):
    nulls = np.array([np.zeros_like(x) for x in xs])
    v_cond_entropy = v_entropy_ensemble_class(xs, models, temps=temps)
    v_null_entropy = v_entropy_ensemble_class(nulls, null_models, temps=null_temps)
    return v_null_entropy - v_cond_entropy