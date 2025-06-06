import numpy as np
import tensorflow as tf
from scipy.stats import norm

import os
import pickle
from tqdm import tqdm

def sample_from_sphere(d):
    vec = np.random.randn(d, 1)
    vec /= np.linalg.norm(vec, axis=0)
    return vec

def psi_bin_train(x, y, n_projs, n_bins):
    n_classes = np.max(y) + 1
    n_samples, n_features = x.shape
    thetas = np.array([sample_from_sphere(n_features) for _ in range(n_projs)])

    psi_bin_data = {
        'n_classes': n_classes,
        'bins': [],
        'joint_probs': [],
        'x_marginal_probs': [],
        'y_marginal_probs': [],
        'thetas': thetas,
    }
    
    unique_y, y_indices = np.unique(y, return_inverse=True)

    for i, theta in enumerate(thetas):
        thetax = np.dot(x, theta)
        bins = np.histogram_bin_edges(thetax, bins=n_bins)
        n_bins = len(bins) - 1
        thetax_binned = np.digitize(thetax, bins) - 1
        thetax_binned = np.clip(thetax_binned, 0, n_bins - 1)
        
        joint_counts = np.zeros((n_bins, len(unique_y)))
        for j in range(len(thetax)):
            joint_counts[thetax_binned[j], y_indices[j]] += 1
        total_count = len(thetax)
        
        joint_probs = joint_counts / total_count
        x_marginal_probs = np.sum(joint_probs, axis=1)
        y_marginal_probs = np.sum(joint_probs, axis=0)
        
        psi_bin_data['bins'].append(bins)
        psi_bin_data['joint_probs'].append(joint_probs)
        psi_bin_data['x_marginal_probs'].append(x_marginal_probs)
        psi_bin_data['y_marginal_probs'].append(y_marginal_probs)

    return psi_bin_data

def psi_bin_val(x, y, psi_bin_data):
    n_projs = len(psi_bin_data['thetas'])
    n_samples = x.shape[0]
    n_classes = psi_bin_data['n_classes']

    pmi_arr = np.zeros((n_projs, n_samples))

    for i, theta in enumerate(psi_bin_data['thetas']):
        thetax = np.dot(x, theta)
        bins = psi_bin_data['bins'][i]
        n_bins = len(bins) - 1
        binned_thetax = np.clip(np.digitize(thetax, bins) - 1, 0, n_bins - 1).squeeze()

        joint_probs = psi_bin_data['joint_probs'][i]
        x_marginal_probs = psi_bin_data['x_marginal_probs'][i]
        y_marginal_probs = psi_bin_data['y_marginal_probs'][i]

        pmi = np.log(joint_probs / (x_marginal_probs[:, None] * y_marginal_probs + 1e-9))
        pmi[joint_probs == 0] = 0
        pmi_arr[i, :] = pmi[binned_thetax, y]
    
    psi = np.mean(pmi_arr, axis=0)

    return psi, pmi_arr

def psi_bin_val_class(x, psi_bin_data):
    n_projs = len(psi_bin_data['thetas'])
    n_samples = x.shape[0]
    n_classes = psi_bin_data['n_classes']

    pmi_arr = np.zeros((n_projs, n_samples, n_classes))

    for i, theta in enumerate(psi_bin_data['thetas']):
        thetax = np.dot(x, theta)
        bins = psi_bin_data['bins'][i]
        n_bins = len(bins) - 1
        binned_thetax = np.clip(np.digitize(thetax, bins) - 1, 0, n_bins - 1).squeeze()

        joint_probs = psi_bin_data['joint_probs'][i]
        x_marginal_probs = psi_bin_data['x_marginal_probs'][i]
        y_marginal_probs = psi_bin_data['y_marginal_probs'][i]

        pmi = np.log(joint_probs / (x_marginal_probs[:, None] * y_marginal_probs + 1e-9))
        pmi[joint_probs == 0] = 0
        pmi_arr[i, np.arange(n_samples)[:, None], np.arange(n_classes)] = pmi[binned_thetax, :]
    
    psi = np.mean(pmi_arr, axis=0)

    return psi, pmi_arr

def psi_gaussian_train(x, y, n_projs):
    n_classes = np.max(y) + 1
    n_samples, n_features = x.shape
    thetas = np.array([sample_from_sphere(n_features) for _ in range(n_projs)])

    psi_gaussian_data = {
        'n_classes': n_classes,
        'thetas': thetas,
        'means': [],
        'std_devs': [],
        'class_means': [],
        'class_std_devs': []
    }
    
    unique_y, y_indices = np.unique(y, return_inverse=True)

    for i, theta in tqdm(enumerate(thetas), desc='Projections'):
        thetax = np.dot(x, theta)
        
        overall_mean = np.mean(thetax)
        overall_std = np.std(thetax)
        
        class_means = []
        class_std_devs = []
        
        for class_label in unique_y:
            class_thetax = thetax[y_indices == class_label]
            class_mean = np.mean(class_thetax)
            class_std = np.std(class_thetax)
            class_means.append(class_mean)
            class_std_devs.append(class_std)
        
        psi_gaussian_data['means'].append(overall_mean)
        psi_gaussian_data['std_devs'].append(overall_std)
        psi_gaussian_data['class_means'].append(class_means)
        psi_gaussian_data['class_std_devs'].append(class_std_devs)

    return psi_gaussian_data

def psi_gaussian_val(x, y, psi_gaussian_data):
    n_projs = len(psi_gaussian_data['thetas'])
    n_samples = x.shape[0]
    n_classes = psi_gaussian_data['n_classes']

    pmi_arr = np.zeros((n_projs, n_samples, n_classes))
    epsilon = 1e-9  # Small value to avoid log of zero and division by zero
    
    for i, theta in tqdm(enumerate(psi_gaussian_data['thetas']), desc='Projections'):
        thetax = np.dot(x, theta)
        
        overall_mean = psi_gaussian_data['means'][i]
        overall_std = psi_gaussian_data['std_devs'][i]
        
        class_means = psi_gaussian_data['class_means'][i]
        class_std_devs = psi_gaussian_data['class_std_devs'][i]
        
        overall_density = norm.pdf(thetax, overall_mean, overall_std) + epsilon
        
        for j, class_label in enumerate(range(n_classes)):
            class_density = norm.pdf(thetax, class_means[j], class_std_devs[j]) + epsilon
            pmi = np.log(class_density / overall_density)
            pmi_arr[i, :, j] = pmi.flatten()
    
    psi_class = np.mean(pmi_arr, axis=0)
    psi = np.array([psi_value[y_val] for psi_value, y_val in zip(psi_class, y)])

    return psi, pmi_arr

def psi_gaussian_val_class(x, psi_gaussian_data):
    n_projs = len(psi_gaussian_data['thetas'])
    n_samples = x.shape[0]
    n_classes = psi_gaussian_data['n_classes']

    pmi_arr = np.zeros((n_projs, n_samples, n_classes))
    epsilon = 1e-9  # Small value to avoid log of zero and division by zero

    for i, theta in tqdm(enumerate(psi_gaussian_data['thetas']), desc='Projections'):
        thetax = np.dot(x, theta)
        
        overall_mean = psi_gaussian_data['means'][i]
        overall_std = psi_gaussian_data['std_devs'][i]
        
        class_means = psi_gaussian_data['class_means'][i]
        class_std_devs = psi_gaussian_data['class_std_devs'][i]
        
        overall_density = norm.pdf(thetax, overall_mean, overall_std) + epsilon
        
        for j, class_label in enumerate(range(n_classes)):
            class_density = norm.pdf(thetax, class_means[j], class_std_devs[j]) + epsilon
            pmi = np.log(class_density / overall_density)
            pmi_arr[i, :, j] = pmi.flatten()
    
    psi = np.mean(pmi_arr, axis=0)

    return psi, pmi_arr