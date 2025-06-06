import numpy as np
import pandas as pd
from scipy.stats import sem, t
import matplotlib.pyplot as plt

from src.metrics import compute_auroc, compute_aurc

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def format_ci(values, scale=1.0):
    values = [v for v in values if v is not None]

    if len(values) == 0:
        return "N/A"

    mean = np.mean(values, axis=0) * scale
    ci = t.ppf(0.975, len(values)-1) * sem(values, axis=0) * scale
    return f"{mean:.2f} ({ci:.2f})"

#######################################################
# Plotting
#######################################################

def plot_histogram_reliability(conf, pred, true, conf_bin_num=1):
    df = pd.DataFrame({'true': true, 'conf': conf, 'pred': pred})
    df['correct'] = (df.pred == df.true).astype(int)
    bins = np.linspace(0, 1, conf_bin_num + 1)
    df['bin'] = pd.cut(df['conf'], bins=bins, include_lowest=True, labels=False)

    bin_acc = df.groupby('bin')['correct'].mean()
    bin_counts = df.groupby('bin')['conf'].count()

    # Bin centers for bar placement
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Fill missing bins with NaNs
    acc = np.zeros(conf_bin_num)
    for i in range(conf_bin_num):
        acc[i] = bin_acc[i] if i in bin_acc else 0

    plt.figure(figsize=(7, 5))
    plt.bar(bin_centers, acc, width=1.0/conf_bin_num, edgecolor='black', align='center', alpha=0.7, label='Model accuracy')
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title('Histogram-Style Reliability Diagram')
    plt.grid(True)
    plt.legend()
    plt.ylim(0, 1)
    plt.xlim(0, 1)
    plt.show()
    
def plot_ece_contributions(conf, pred, true, conf_bin_num=10):
    """
    Plots a bar chart of ECE contributions per bin.
    """
    df = pd.DataFrame({'true': true, 'conf': conf, 'pred': pred})
    df['correct'] = (df.pred == df.true).astype(int)
    bins = np.linspace(0, 1, conf_bin_num + 1)
    df['bin'] = pd.cut(df['conf'], bins=bins, include_lowest=True, labels=False)

    # Bin-wise stats
    bin_acc = df.groupby('bin')['correct'].mean()
    bin_conf = df.groupby('bin')['conf'].mean()
    bin_counts = df.groupby('bin')['conf'].count()
    total = len(df)

    # Bin centers
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Compute ECE contributions
    bin_ece_contrib = np.zeros(conf_bin_num)
    for i in range(conf_bin_num):
        if i in bin_acc and i in bin_conf:
            bin_ece_contrib[i] = np.abs(bin_acc[i] - bin_conf[i]) * (bin_counts[i] / total)

    # Plot
    plt.figure(figsize=(7, 4))
    plt.bar(bin_centers, bin_ece_contrib, width=1.0/conf_bin_num, edgecolor='black', align='center', alpha=0.7)
    plt.xlabel('Confidence')
    plt.ylabel('ECE Contribution')
    plt.title('Per-Bin Contribution to ECE')
    plt.grid(True)
    plt.show()
    
def plot_bin_counts(conf, conf_bin_num=10):
    bins = np.linspace(0, 1, conf_bin_num + 1)
    bin_indices = pd.cut(conf, bins=bins, include_lowest=True, labels=False)

    # Ensure we get counts for all bins, even if some bins have 0
    counts = pd.Series(bin_indices).value_counts().sort_index()
    full_counts = np.zeros(conf_bin_num)
    for i in range(conf_bin_num):
        if i in counts:
            full_counts[i] = counts[i]

    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Now shapes match: both are (conf_bin_num,)
    plt.figure(figsize=(7, 3))
    plt.bar(bin_centers, full_counts, width=1.0/conf_bin_num, edgecolor='black', align='center')
    plt.xlabel('Confidence')
    plt.ylabel('Sample Count')
    plt.title('Number of Samples per Confidence Bin')
    plt.grid(True)
    plt.show()


#######################################################
# Filtering
#######################################################

def compute_opt_threshold(metric, true_label):
    sorted_metric = np.sort(metric)
    best_threshold = 0.0
    best_acc = 0.0
    for threshold in sorted_metric:
        metric_label = (metric >= threshold).astype(int)
        correct_samples = np.sum(true_label == metric_label)
        filtering_acc = (correct_samples / len(true_label)) * 100
        if filtering_acc > best_acc:
            best_acc = filtering_acc
            best_threshold = threshold
    return best_threshold

def compute_filtering_pr(metric, true_label, threshold):
    pred_labels = (metric <= threshold).astype(int)
    true_positives = np.sum((pred_labels == 1) & (true_label == 1))
    false_positives = np.sum((pred_labels == 1) & (true_label == 0))
    false_negatives = np.sum((pred_labels == 0) & (true_label == 1))
    precision = true_positives / (true_positives + false_positives) * 100 if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) * 100 if (true_positives + false_negatives) > 0 else 0
    return precision, recall

def reliability_diagram(metric, true_y, pred_y, n_bins):
    bin_size = 1.0 / n_bins
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    indices = np.digitize(metric, bins, right=True)

    bin_acc = np.zeros(n_bins, dtype=np.float32)
    bin_conf = np.zeros(n_bins, dtype=np.float32)
    bin_counts = np.zeros(n_bins, dtype=np.int32)

    for b in range(n_bins):
        selected = np.where(indices == b + 1)[0]
        if len(selected) > 0:
            bin_acc[b] = np.mean(true_y[selected] == pred_y[selected])
            bin_conf[b] = np.mean(metric[selected])
            bin_counts[b] = len(selected)

    avg_acc = np.sum(bin_acc * bin_counts) / np.sum(bin_counts)
    avg_conf = np.sum(bin_conf * bin_counts) / np.sum(bin_counts)

    gaps = np.abs(bin_acc - bin_conf)
    ece = np.sum(gaps * bin_counts) / np.sum(bin_counts) * 100
    
    result = {
        'bins' : bins,
        'bin_size' : bin_size,
        'bin_counts' : bin_counts,
        'bin_acc' : bin_acc,
        'bin_conf' : bin_conf,
        'avg_acc' : avg_acc,
        'avg_conf' : avg_conf,
        'gaps' : gaps,
        'ece' : ece
    }
    
    return result

def plot_reliability_diagram(result, metric):
    positions = result['bins'][:-1] + result['bin_size']/2.0

    fig, axs = plt.subplots(2, 1, figsize=(4,5), dpi=100, sharex=True, gridspec_kw={'height_ratios': [1,0.5]})

    axs[0].bar(positions,
            result['bin_acc'],
            width=result['bin_size'],
            edgecolor='black',
            color='blue',
            linewidth=1,
            label='Accuracy')

    axs[0].bar(positions,
            result['gaps'],
            bottom=np.minimum(result['bin_acc'], result['bin_conf']),
            width=result['bin_size'],
            edgecolor='black',
            color='red',
            linewidth=1,
            hatch="//",
            label='Gap')

    axs[0].text(0.7, 0.1,
                f"ECE={result['ece']:.2f}",
                color="black",
                bbox=dict(facecolor='white', alpha=0.5),
                fontsize=11)

    axs[0].plot([0,1], [0,1], linestyle = "--", color="gray")
    axs[0].set_xlim(0,1)
    axs[0].grid(True, alpha=0.3)
    axs[0].grid(zorder=0)
    axs[0].set_ylabel('Accuracy', fontsize=11)
    axs[0].legend()
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    axs[1].bar(positions,
               result['bin_counts'] / np.sum(result['bin_counts']),
               width=result['bin_size'],
               edgecolor='black',
               color='tab:orange')

    axs[1].axvline(x=result['avg_acc'], linestyle="--", color="blue", label='Average Accuracy')
    axs[1].axvline(x=result['avg_conf'], linestyle='--', color='red', label='Average Confidence')

    axs[1].grid(True)
    axs[1].set_xticks(np.linspace(0.0, 1.0, 11))
    axs[1].set_xlabel(metric, fontsize=11)
    axs[1].set_ylabel('% of Samples', fontsize=11)
    axs[1].legend()

    plt.tight_layout()
    plt.show()

def compute_classification_error(metric, true_y):
    count = 0
    for i in range(len(metric)):
        pred = np.argmax(metric[i])
        if true_y[i] == pred:
            count += 1
    error = 1 - count/len(metric)
    return error

def compute_selective_pred_acc(metric, dataset, model, n):
    x = np.array([x for x, y in dataset])
    y = np.array([y for x, y in dataset])
    idx = np.argsort(metric)[n:]
    eval_ds = model.evaluate(x[idx], y[idx], verbose=0)
    return eval_ds[1]

def calculate_confidence_interval(data, confidence=0.95):
    n = data.shape[0]
    mean = np.mean(data, axis=0)
    se = np.std(data, axis=0) / np.sqrt(n)
    h = se * 1.96  # For 95% confidence interval
    return mean, h

def create_percentile_range(min_value, max_value, percentiles=100):
    step = (max_value - min_value) / (percentiles + 1)
    percentile_values = [min_value + step * i for i in range(1, percentiles + 1)]
    return percentile_values

