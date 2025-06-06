import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from scipy.stats import pearsonr, spearmanr, kendalltau

#-------------------------------
# Metrics for Failure Prediction
#-------------------------------

def compute_auroc(conf, pred, true):
    """
    Area Under the ROC Curve (AUROC) for correct predictions.

    This metric evaluates how well the confidence scores distinguish between
    correct and incorrect predictions.

    Args:
        conf (np.ndarray): List of confidences for top predicted class
        pred (np.ndarray): List of predictions
        true (np.ndarray): List of true labels

    Returns:
        auroc (float): AUROC score. Higher indicates better separation between correct and incorrect predictions.
    """
    
    success_label = np.equal(true, pred).astype(int)
    auroc = roc_auc_score(success_label, conf)
    return auroc

def compute_auprc_success(conf, pred, true):
    """
    Area Under the Precision-Recall Curve (AUPRC) for correct predictions.

    This metric evaluates how well the model's confidence scores correspond to correct predictions.
    It treats correct predictions as the positive class and computes the average precision accordingly.

    Args:
        conf (np.ndarray): List of confidences for top predicted class
        pred (np.ndarray): List of predictions
        true (np.ndarray): List of true labels

    Returns:
        auprc_success (float): AUPRC for successes. Higher indicates better alignment between confidence and correctness.
    """
    success_label = np.equal(true, pred).astype(int)
    auprc_success = average_precision_score(success_label, conf, pos_label=1)
    return auprc_success

def compute_auprc_error(conf, pred, true):
    """
    Area Under the Precision-Recall Curve (AUPRC) for incorrect predictions.

    This metric evaluates how well the confidence scores can identify incorrect predictions.
    It treats errors (i.e. incorrect predictions) as the positive class and computes the
    average precision score with respect to them.

    Args:
        conf (np.ndarray): List of confidences for top predicted class
        pred (np.ndarray): List of predictions
        true (np.ndarray): List of true labels
        
    Returns:
        auprc_error (float): AUPRC for errors. Higher indicates better ability to detect incorrect predictions.
    """
    
    success_label = np.equal(true, pred).astype(int)
    auprc_error = average_precision_score(success_label, -conf, pos_label=0)
    return auprc_error

def compute_aurc(conf, pred, true):
    """
    Area Under the Risk-Coverage Curve (AURC).

    AURC measures how well a model’s confidence scores rank predictions
    by their correctness. Lower AURC indicates better calibration and ranking performance.

    Args:
        conf (np.ndarray): List of confidences for top predicted class
        pred (np.ndarray): List of predictions
        true (np.ndarray): List of true labels

    Returns:
        aurc (float): AURC value. Lower is better.
    """
    
    success_label = np.equal(true, pred).astype(int)
    
    coverages = []
    risks = []
    residuals = 1 - success_label
    n = len(residuals)
    idx_sorted = np.argsort(conf)
    cov = n
    error_sum = sum(residuals[idx_sorted])
    coverages.append(cov/ n)
    risks.append(error_sum / n)
    weights = []
    tmp_weight = 0
    
    for i in range(0, len(idx_sorted) - 1):
        cov = cov-1
        error_sum = error_sum - residuals[idx_sorted[i]]
        selective_risk = error_sum /(n - 1 - i)
        tmp_weight += 1
        if i == 0 or conf[idx_sorted[i]] != conf[idx_sorted[i - 1]]:
            coverages.append(cov / n)
            risks.append(selective_risk)
            weights.append(tmp_weight / n)
            tmp_weight = 0
            
    if tmp_weight > 0:
        coverages.append(0)
        risks.append(risks[-1])
        weights.append(tmp_weight / n)
        
    aurc = sum([(risks[i] + risks[i+1]) * 0.5 * weights[i] for i in range(len(weights)) ])
    return aurc

def compute_eaurc(conf, pred, true):
    """
    Expected Area Under the Risk-Coverage Curve (EAURC).

    EAURC measures how well a model’s confidence scores rank predictions
    by their correctness, normalized against the ideal performance (oracle risk).
    Lower EAURC indicates better calibration and ranking performance.

    Args:
        conf (np.ndarray): List of confidences for top predicted class
        pred (np.ndarray): List of predictions
        true (np.ndarray): List of true labels

    Returns:
        eaurc (float): EAURC value. Lower is better.
    """
    
    success_label = np.equal(true, pred).astype(int)
    
    coverages = []
    risks = []
    residuals = 1 - success_label  # 1 if incorrect, 0 if correct
    n = len(residuals)
    idx_sorted = np.argsort(conf)
    cov = n
    error_sum = sum(residuals[idx_sorted])
    coverages.append(cov / n)
    risks.append(error_sum / n)
    weights = []
    tmp_weight = 0

    for i in range(0, len(idx_sorted) - 1):
        cov -= 1
        error_sum -= residuals[idx_sorted[i]]
        selective_risk = error_sum / (n - 1 - i)
        tmp_weight += 1
        if i == 0 or conf[idx_sorted[i]] != conf[idx_sorted[i - 1]]:
            coverages.append(cov / n)
            risks.append(selective_risk)
            weights.append(tmp_weight / n)
            tmp_weight = 0

    if tmp_weight > 0:
        coverages.append(0)
        risks.append(risks[-1])
        weights.append(tmp_weight / n)

    aurc = sum([(risks[i] + risks[i + 1]) * 0.5 * weights[i] for i in range(len(weights))])

    accuracy = np.mean(success_label)
    if accuracy > 0:
        optimal_risk = (1 - accuracy) + accuracy * np.log(accuracy)
    else:
        optimal_risk = 1.0  # worst case
    eaurc = aurc - optimal_risk
    return eaurc

def compute_naurc(conf, pred, true):
    """
    Computes the Normalized AURC (NAURC)

    Args:
        conf (np.ndarray): Confidence scores
        pred (np.ndarray): Model predictions
        true (np.ndarray): Ground truth labels

    Returns:
        naurc (float): Normalized AURC value
    """
    eaurc = compute_eaurc(conf, pred, true)
    
    success_label = np.equal(true, pred).astype(int)
    accuracy = np.mean(success_label)
    if accuracy > 0:
        aurc_ideal = (1 - accuracy) + accuracy * np.log(accuracy)
    else:
        aurc_ideal = 1.0
        
    risk = 1 - accuracy
    
    denom = risk - aurc_ideal
    if denom == 0:
        return 0.0
    
    naurc = eaurc / denom
    return naurc

def compute_fpr_at_95tpr(conf, pred, true, tolerance=0.0005, resolution=10000):
    """
    FPR when TPR is approximately 95%, within a given tolerance.
    
    Args:
        conf (np.ndarray): List of confidences for top predicted class
        pred (np.ndarray): List of predictions
        true (np.ndarray): List of true labels
        tolerance (float): Allowed deviation from 0.95 TPR. Default is ±0.0005.
        resolution (int): Number of thresholds to scan. Higher = more precise.

    Returns:
        fpr (float): False Positive Rate at ~95% TPR.
    """
    
    success_label = np.equal(true, pred).astype(int)
    accurate = success_label == 1
    errors = success_label == 0

    thresholds = np.linspace(conf.min(), conf.max(), resolution)

    for delta in thresholds:
        selected = conf >= delta

        if selected.sum() == 0:
            continue

        tpr = np.sum(accurate & selected) / np.sum(accurate)
        if 0.95 - tolerance <= tpr <= 0.95 + tolerance:
            fpr = np.sum(errors & selected) / np.sum(errors)
            return fpr

    # If no threshold satisfies the condition
    return None

#------------------------------------
# Metrics for Correlation with Margin
#------------------------------------
    
def spearman_corr(metric, margin):
    corr, _ = spearmanr(metric, margin)
    return corr

def pearson_corr(metric,margin):
    corr, _ = pearsonr(metric, margin)
    return corr

def kendall_corr(metric, margin):
    corr, _ = kendalltau(metric, margin)
    return corr

#-----------------------------------
# Metrics for Confidence Calibration
#-----------------------------------

def compute_acc_bin(conf_thresh_lower, conf_thresh_upper, conf, pred, true):
    """
    Computes accuracy and average confidence for bin
    
    Args:
        conf_thresh_lower (float): Lower Threshold of confidence interval
        conf_thresh_upper (float): Upper Threshold of confidence interval
        conf (np.ndarray): List of confidences for top predicted class
        pred (np.ndarray): List of predictions
        true (np.ndarray): List of true labels
    
    Returns:
        (accuracy, avg_conf, len_bin): accuracy of bin, confidence of bin and number of elements in bin.
    """
    filtered_tuples = [x for x in zip(pred, true, conf) if x[2] > conf_thresh_lower and x[2] <= conf_thresh_upper]
    if len(filtered_tuples) < 1:
        return 0,0,0
    else:
        correct = len([x for x in filtered_tuples if x[0] == x[1]])  
        len_bin = len(filtered_tuples)  
        avg_conf = sum([x[2] for x in filtered_tuples]) / len_bin  
        accuracy = float(correct)/len_bin 
        return accuracy, avg_conf, len_bin
    
def compute_ece(conf, pred, true, conf_bin_num = 10, threshold=0.01):
    """
    Expected Calibration Error (ECE)
    
    ECE measures how well the model's predicted confidence scores match the actual
    accuracy.
    
    Args:
        conf (np.ndarray): List of confidences for top predicted class
        pred (np.ndarray): List of predictions
        true (np.ndarray): List of true labels
        conf_bin_num: (float): Number of equal-width bins to divide the [0, 1] confidence range into. Default is 10.
        threshold (float): Minimum confidence to include in the calculation. Default is 0.01.
        
    Returns:
        ece (float): ECE value. Lower is better.
    """
    df = pd.DataFrame({'ys':true, 'conf':conf, 'pred':pred})
    df = df[df['conf'] >= threshold]
    if len(df) == 0:
        return 0.0
    df['correct'] = (df.pred == df.ys).astype('int')
    
    bin_bounds = np.linspace(0, 1, conf_bin_num + 1)[1:-1]
    df['conf_bin'] = pd.cut(df['conf'], bins=np.linspace(0.0, 1.0, conf_bin_num + 1), include_lowest=True, labels=False)

    group_acc = df.groupby(['conf_bin'])['correct'].mean()
    group_confs = df.groupby(['conf_bin'])['conf'].mean()
    counts = df.groupby(['conf_bin'])['conf'].count()
    ece = (np.abs(group_acc - group_confs) * counts / len(df)).sum()
        
    return ece

def compute_cc_ece(conf, pred, true, conf_bin_num=10, threshold=0.01):
    """
    Class-Conditional Expected Calibration Error (CC-ECE)

    Args:
        conf (np.ndarray): Confidence scores for the top predicted class.
        pred (np.ndarray): Predicted class labels.
        true (np.ndarray): True class labels.
        conf_bin_num (int): Number of confidence bins.

    Returns:
        cc_ece (float): Class-Conditional ECE.
    """
    df = pd.DataFrame({'conf': conf, 'pred': pred, 'true': true})
    df = df[df['conf'] >= threshold]
    if len(df) == 0:
        return 0.0
    df['correct'] = (df['pred'] == df['true']).astype(int)
    
    class_ece_list = []

    for cls in np.unique(pred):
        cls_df = df[df['pred'] == cls]
        if len(cls_df) == 0:
            continue  # skip empty classes

        cls_df = cls_df.copy()
        cls_df['conf_bin'] = pd.cut(cls_df['conf'], bins=np.linspace(0.0, 1.0, conf_bin_num + 1), include_lowest=True, labels=False)

        group_acc = cls_df.groupby('conf_bin')['correct'].mean()
        group_conf = cls_df.groupby('conf_bin')['conf'].mean()
        bin_counts = cls_df.groupby('conf_bin')['conf'].count()

        cls_ece = (np.abs(group_acc - group_conf) * bin_counts / len(cls_df)).sum()
        class_ece_list.append(cls_ece)

    cc_ece = np.mean(class_ece_list)
    return cc_ece

def compute_mce(conf, pred, true, conf_bin_num = 10, threshold=0.01):

    """
    Maximal Calibration Error (MCE)
    
    MCE captures the largest absolute difference between confidence and accuracy
    across all confidence bins.
    
    Args:
        conf (np.ndarray): List of confidences for top predicted class
        pred (np.ndarray): List of predictions
        true (np.ndarray): List of true labels
        conf_bin_num: (float): Number of equal-width bins to divide the [0, 1] confidence range into. Default is 10.
        
    Returns:
        mce (float): MCE value. Lower is better.
    """
    df = pd.DataFrame({'ys':true, 'conf':conf, 'pred':pred})
    df = df[df['conf'] >= threshold]
    if len(df) == 0:
        return 0.0
    df['correct'] = (df.pred == df.ys).astype('int')

    bin_bounds = np.linspace(0, 1, conf_bin_num + 1)[1:-1]
    df['conf_bin'] = df['conf'].apply(lambda x: np.digitize(x, bin_bounds))

    group_acc = df.groupby(['conf_bin'])['correct'].mean()
    group_confs = df.groupby(['conf_bin'])['conf'].mean()
    counts = df.groupby(['conf_bin'])['conf'].count()
    mce = (np.abs(group_acc - group_confs) * counts / len(df)).max()
        
    return mce

def compute_ace(conf, pred, true, conf_bin_num=10, threshold=0.01):
    """
    Average Calibration Error (ACE)
    
    ACE measures the average absolute difference between confidence and accuracy
    across all confidence bins, treating each bin equally (unlike ECE, which weights by bin occupancy).
    
    Args:
        conf (np.ndarray): List of confidences for top predicted class
        pred (np.ndarray): List of predictions
        true (np.ndarray): List of true labels
        conf_bin_num: (float): Number of equal-width bins to divide the [0, 1] confidence range into. Default is 10.
        
    Returns:
        ace (float): ACE value. Lower is better.
    """
    
    df = pd.DataFrame({'ys': true, 'conf': conf, 'pred': pred})
    df = df[df['conf'] >= threshold]
    if len(df) == 0:
        return 0.0
    df['correct'] = (df.pred == df.ys).astype(int)
    
    bin_bounds = np.linspace(0, 1, conf_bin_num + 1)[1:-1]
    df['conf_bin'] = df['conf'].apply(lambda x: np.digitize(x, bin_bounds))
    
    group_acc = df.groupby('conf_bin')['correct'].mean()
    group_confs = df.groupby('conf_bin')['conf'].mean()
    
    ace = np.abs(group_acc - group_confs).mean()
    
    return ace

def compute_adaece(conf, pred, true, conf_bin_num=10, threshold=0.01):

    """
    Adaptive Expected Calibration Error (Ada-ECE)
    
    Ada-ECE is a variant of Expected Calibration Error (ECE) that uses 
    quantile-based (equal-frequency) binning to ensure that each bin contains 
    approximately the same number of samples.
    
    Args:
        conf (np.ndarray): List of confidences for top predicted class
        pred (np.ndarray): List of predictions
        true (np.ndarray): List of true labels
        conf_bin_num: (float): Number of quantile bins (equal-frequency). Default is 10.
        
    Returns:
        adaece: AdaECE value. Lower is better.
    """
    df = pd.DataFrame({'ys':true, 'conf':conf, 'pred':pred})
    df = df[df['conf'] >= threshold]
    if len(df) == 0:
        return 0.0
    df['correct'] = (df.pred == df.ys).astype('int')
    df['conf_bin'] = pd.qcut(df['conf'], q=conf_bin_num, duplicates='drop', labels=False)

    group_acc = df.groupby('conf_bin')['correct'].mean()
    group_confs = df.groupby('conf_bin')['conf'].mean()
    counts = df.groupby('conf_bin')['conf'].count()
    
    ada_ece = (np.abs(group_acc - group_confs) * counts / len(df)).sum()
    return ada_ece

def compute_cc_adaece(conf, pred, true, conf_bin_num=10, threshold=0.01):
    """
    Class-Conditional Adaptive Expected Calibration Error (CC-AdaECE)

    Args:
        conf (np.ndarray): Confidence scores for top predicted class.
        pred (np.ndarray): Predicted class labels.
        true (np.ndarray): True class labels.
        conf_bin_num (int): Number of quantile bins (equal-frequency).

    Returns:
        cc_adaece (float): Class-Conditional AdaECE.
    """
    df = pd.DataFrame({'conf': conf, 'pred': pred, 'true': true})
    df = df[df['conf'] >= threshold]
    if len(df) == 0:
        return 0.0
    df['correct'] = (df['pred'] == df['true']).astype(int)
    
    class_adaece_list = []

    for cls in np.unique(pred):
        cls_df = df[df['pred'] == cls]
        if len(cls_df) < 2:
            continue 

        cls_df = cls_df.copy()
        try:
            cls_df['conf_bin'] = pd.qcut(cls_df['conf'], q=conf_bin_num, duplicates='drop', labels=False)
        except ValueError:
            continue  

        group_acc = cls_df.groupby('conf_bin')['correct'].mean()
        group_conf = cls_df.groupby('conf_bin')['conf'].mean()
        bin_counts = cls_df.groupby('conf_bin')['conf'].count()

        cls_adaece = (np.abs(group_acc - group_conf) * bin_counts / len(cls_df)).sum()
        class_adaece_list.append(cls_adaece)

    cc_adaece = np.mean(class_adaece_list) if class_adaece_list else 0.0
    return cc_adaece

def compute_sce(conf_class, true, num_classes, conf_bin_num=10):
    """
    Static Calibration Error (SCE).

    Bins predicted probabilities for each class independently, compares the 
    average predicted probability to the empirical accuracy in each bin, 
    and aggregates the weighted error across all bins and classes.

    Args:
        conf_class (np.ndarray): Shape (N, C), predicted class probabilities
        true (np.ndarray): Shape (N,), ground truth labels
        num_classes (int): Total number of classes
        conf_bin_num (int): Number of equal-width bins (default: 10)

    Returns:
        float: Static Calibration Error (SCE)
    """
    N = len(true)
    total_error = 0.0

    for k in range(num_classes):
        p_k = conf_class[:, k]
        y_k = (true == k).astype(int)
        
        df = pd.DataFrame({'p_k': p_k, 'y_k': y_k})
        bin_bounds = np.linspace(0.0, 1.0, conf_bin_num + 1)
        df['bin'] = pd.cut(df['p_k'], bins=bin_bounds, include_lowest=True, labels=False)

        group_acc = df.groupby('bin')['y_k'].mean()
        group_conf = df.groupby('bin')['p_k'].mean()
        counts = df.groupby('bin')['p_k'].count()

        for b in group_acc.index:
            acc = group_acc[b]
            conf = group_conf[b]
            count = counts[b]
            total_error += (count / N) * abs(acc - conf)

    sce = total_error / num_classes
    return sce

def compute_adasce(conf_class, true, num_classes, conf_bin_num=10, threshold=0.01):
    """
    Adaptive Static Calibration Error (AdaSCE).

    Computes SCE with quantile-based binning for each class.

    Args:
        conf_class (np.ndarray): Shape (N, C), predicted class probabilities
        true (np.ndarray): Shape (N,), ground truth labels
        num_classes (int): Total number of classes
        conf_bin_num (int): Number of quantile bins per class (default: 10)
        threshold (float): Ignore predictions with p_k below this value

    Returns:
        float: Adaptive Static Calibration Error (AdaSCE)
    """
    N = len(true)
    total_error = 0.0

    for k in range(num_classes):
        p_k = conf_class[:, k]
        y_k = (true == k).astype(int)

        df = pd.DataFrame({'p_k': p_k, 'y_k': y_k})
        df = df[df['p_k'] >= threshold]  # Optional: ignore low-confidence predictions
        if len(df) == 0:
            continue

        try:
            df['bin'] = pd.qcut(df['p_k'], q=conf_bin_num, duplicates='drop', labels=False)
        except ValueError:
            continue  # Skip if not enough unique values for qcut

        group_acc = df.groupby('bin')['y_k'].mean()
        group_conf = df.groupby('bin')['p_k'].mean()
        counts = df.groupby('bin')['p_k'].count()

        for b in group_acc.index:
            acc = group_acc[b]
            conf = group_conf[b]
            count = counts[b]
            total_error += (count / N) * abs(acc - conf)

    ada_sce = total_error / num_classes
    return ada_sce

def compute_cc_adasce(conf_class, true, num_classes, conf_bin_num=10, threshold=0.01):
    """
    Class-Conditional Adaptive Static Calibration Error (CC-AdaSCE).

    Computes AdaSCE conditioned on the true class, i.e., for each class,
    evaluates how well calibrated the model is when the true label is that class.

    Args:
        conf_class (np.ndarray): Shape (N, C), predicted class probabilities
        true (np.ndarray): Shape (N,), ground truth labels
        num_classes (int): Total number of classes
        conf_bin_num (int): Number of quantile bins (default: 10)
        threshold (float): Minimum confidence to include in calibration (default: 0.01)

    Returns:
        float: CC-AdaSCE value
    """
    N = len(true)
    class_errors = []

    for k in range(num_classes):
        p_k = conf_class[:, k]
        y_k = (true == k).astype(int)

        df = pd.DataFrame({'p_k': p_k, 'y_k': y_k})
        df = df[df['y_k'] == 1]  # Only consider samples where the true label is class k
        df = df[df['p_k'] >= threshold]  # Filter low-confidence

        if len(df) < 2:
            continue

        try:
            df['bin'] = pd.qcut(df['p_k'], q=conf_bin_num, duplicates='drop', labels=False)
        except ValueError:
            continue  # Not enough unique values to bin

        group_acc = df.groupby('bin')['y_k'].mean()  # Will be 1.0 (since all y_k == 1)
        group_conf = df.groupby('bin')['p_k'].mean()
        bin_counts = df.groupby('bin')['p_k'].count()

        class_error = (np.abs(group_acc - group_conf) * bin_counts / N).sum()
        class_errors.append(class_error)

    cc_adasce = np.mean(class_errors) if class_errors else 0.0
    return cc_adasce

def compute_cw_ece(conf_class, true, num_classes, conf_bin_num=10):
    """
    Classwise Expected Calibration Error (CW-ECE).
    
    CW-ECE evaluates the calibration of predicted probabilities for each class 
    individually.

    Args:
        conf_class (np.ndarray): List of confidences for all classes
        true (np.ndarray): List of true labels
        num_classes (int): Total number of classes
        conf_bin_num: (float): Number of equal-width bins to divide the [0, 1] confidence range into. Default is 10.

    Returns:
        cw_ece (float): CW-ECE value. Lower is better.
    """
    
    class_errors = []

    for class_idx in range(num_classes):
        p_i = conf_class[:, class_idx]
        y_i = (true == class_idx).astype(int)

        df = pd.DataFrame({'p_i': p_i, 'y_i': y_i})
        bin_bounds = np.linspace(0, 1, conf_bin_num + 1)[1:-1]
        df['bin'] = df['p_i'].apply(lambda x: np.digitize(x, bin_bounds))

        group_acc = df.groupby('bin')['y_i'].mean()
        group_conf = df.groupby('bin')['p_i'].mean()
        counts = df.groupby('bin')['p_i'].count()
        
        group_acc = group_acc.dropna()
        group_conf = group_conf.dropna()
        counts = counts[group_acc.index]

        ece_i = (np.abs(group_acc - group_conf) * counts / len(df)).sum()
        class_errors.append(ece_i)

    cw_ece = np.mean(class_errors)

    return cw_ece

def compute_cw_adaece(conf_class, true, num_classes, conf_bin_num=10):
    """
    Classwise Adaptive Expected Calibration Error (CW-AdaECE).

    Uses quantile-based binning (equal-frequency bins) within each class to 
    compute calibration error.

    Args:
        conf_class (np.ndarray): Array of shape (N, C), where each entry is the model's predicted probability for each class
        true (np.ndarray): True labels of shape (N,)
        num_classes (int): Total number of classes
        conf_bin_num (int): Number of quantile bins. Default is 10.

    Returns:
        cw_ada_ece (float): Classwise Adaptive ECE value
    """
    class_errors = []

    for class_idx in range(num_classes):
        p_i = conf_class[:, class_idx]
        y_i = (true == class_idx).astype(int)

        df = pd.DataFrame({'p_i': p_i, 'y_i': y_i})
        
        try:
            df['bin'] = pd.qcut(df['p_i'], q=conf_bin_num, duplicates='drop', labels=False)
        except ValueError:
            # Not enough unique values to bin; skip this class
            continue

        group_acc = df.groupby('bin')['y_i'].mean()
        group_conf = df.groupby('bin')['p_i'].mean()
        counts = df.groupby('bin')['p_i'].count()

        ece_i = (np.abs(group_acc - group_conf) * counts / len(df)).sum()
        class_errors.append(ece_i)

    cw_ada_ece = np.mean(class_errors) if class_errors else 0.0
    return cw_ada_ece

def compute_cw_adaece_rms(conf_class, true, num_classes, conf_bin_num=10):
    """
    RMS Classwise Adaptive Expected Calibration Error (CW-AdaECE with L2 norm).

    Args:
        conf_class (np.ndarray): Array of shape (N, C), where each entry is the model's predicted probability for each class.
        true (np.ndarray): True labels of shape (N,)
        num_classes (int): Total number of classes.
        conf_bin_num (int): Number of quantile bins. Default is 10.

    Returns:
        cw_adaece_rms (float): Classwise Adaptive ECE using root mean square error.
    """
    class_errors = []

    for class_idx in range(num_classes):
        p_i = conf_class[:, class_idx]
        y_i = (true == class_idx).astype(int)

        df = pd.DataFrame({'p_i': p_i, 'y_i': y_i})

        try:
            df['bin'] = pd.qcut(df['p_i'], q=conf_bin_num, duplicates='drop', labels=False)
        except ValueError:
            continue  # Skip if not enough unique values

        group_acc = df.groupby('bin')['y_i'].mean()
        group_conf = df.groupby('bin')['p_i'].mean()
        counts = df.groupby('bin')['p_i'].count()

        # Drop bins with NaNs
        valid_idx = group_acc.index.intersection(group_conf.index)
        acc = group_acc.loc[valid_idx]
        conf = group_conf.loc[valid_idx]
        count = counts.loc[valid_idx]

        squared_diff = ((acc - conf) ** 2) * count
        mse = squared_diff.sum() / len(df)
        rms_ece_i = np.sqrt(mse)

        class_errors.append(rms_ece_i)

    cw_adaece_rms = np.mean(class_errors) if class_errors else 0.0
    return cw_adaece_rms

def compute_cw_sce(conf_class, true, num_classes, conf_bin_num=10):
    """
    Classwise Static Calibration Error (CW-SCE).

    CW-SCE evaluates the calibration of predicted probabilities for each class
    individually, normalized by the total number of samples (global normalization).

    Args:
        conf_class (np.ndarray): Array of shape (N, C), predicted probabilities for each class.
        true (np.ndarray): Array of shape (N,), true class labels.
        num_classes (int): Total number of classes.
        conf_bin_num (int): Number of equal-width bins. Default is 10.

    Returns:
        cw_sce (float): CW-SCE value. Lower is better.
    """
    class_errors = []
    N = len(true)  # total number of samples

    for class_idx in range(num_classes):
        p_i = conf_class[:, class_idx]
        y_i = (true == class_idx).astype(int)

        df = pd.DataFrame({'p_i': p_i, 'y_i': y_i})
        df['bin'] = pd.cut(df['p_i'], bins=np.linspace(0, 1, conf_bin_num + 1), labels=False, include_lowest=True)

        group_acc = df.groupby('bin')['y_i'].mean()
        group_conf = df.groupby('bin')['p_i'].mean()
        counts = df.groupby('bin')['p_i'].count()

        valid_idx = group_acc.index.intersection(group_conf.index)
        acc = group_acc.loc[valid_idx]
        conf = group_conf.loc[valid_idx]
        count = counts.loc[valid_idx]

        sce_i = (np.abs(acc - conf) * count / N).sum()
        class_errors.append(sce_i)

    cw_sce = np.mean(class_errors)
    return cw_sce

def compute_cw_adasce(conf_class, true, num_classes, conf_bin_num=10):
    """
    Classwise Adaptive Static Calibration Error (CW-AdaSCE).

    Uses quantile-based binning within each class and normalizes over the total number of samples.

    Args:
        conf_class (np.ndarray): Array of shape (N, C), model-predicted probabilities per class.
        true (np.ndarray): Array of shape (N,), true labels.
        num_classes (int): Number of classes.
        conf_bin_num (int): Number of quantile bins. Default is 10.

    Returns:
        cw_ada_sce (float): Classwise Adaptive Static Calibration Error.
    """
    class_errors = []
    N = len(true)  # total number of samples

    for class_idx in range(num_classes):
        p_i = conf_class[:, class_idx]
        y_i = (true == class_idx).astype(int)

        df = pd.DataFrame({'p_i': p_i, 'y_i': y_i})

        try:
            df['bin'] = pd.qcut(df['p_i'], q=conf_bin_num, duplicates='drop', labels=False)
        except ValueError:
            continue  # skip this class if not enough unique values

        group_acc = df.groupby('bin')['y_i'].mean()
        group_conf = df.groupby('bin')['p_i'].mean()
        counts = df.groupby('bin')['p_i'].count()

        valid_idx = group_acc.index.intersection(group_conf.index)
        acc = group_acc.loc[valid_idx]
        conf = group_conf.loc[valid_idx]
        count = counts.loc[valid_idx]

        sce_i = (np.abs(acc - conf) * count / N).sum()
        class_errors.append(sce_i)

    cw_ada_sce = np.mean(class_errors) if class_errors else 0.0
    return cw_ada_sce

def compute_cw_ada_sce_rms(conf_class, true, num_classes, conf_bin_num=10):
    """
    RMS Classwise Adaptive Static Calibration Error (CW-AdaSCE with L2 norm).

    Args:
        conf_class (np.ndarray): Array of shape (N, C), model-predicted probabilities per class.
        true (np.ndarray): Array of shape (N,), true labels.
        num_classes (int): Number of classes.
        conf_bin_num (int): Number of quantile bins. Default is 10.

    Returns:
        cw_ada_sce_rms (float): RMS of Classwise Adaptive Static Calibration Error.
    """
    class_errors = []
    N = len(true)  # total number of samples (global normalization)

    for class_idx in range(num_classes):
        p_i = conf_class[:, class_idx]
        y_i = (true == class_idx).astype(int)

        df = pd.DataFrame({'p_i': p_i, 'y_i': y_i})

        try:
            df['bin'] = pd.qcut(df['p_i'], q=conf_bin_num, duplicates='drop', labels=False)
        except ValueError:
            continue  # Skip if not enough unique values

        group_acc = df.groupby('bin')['y_i'].mean()
        group_conf = df.groupby('bin')['p_i'].mean()
        counts = df.groupby('bin')['p_i'].count()

        valid_idx = group_acc.index.intersection(group_conf.index)
        acc = group_acc.loc[valid_idx]
        conf = group_conf.loc[valid_idx]
        count = counts.loc[valid_idx]

        squared_diff = ((acc - conf) ** 2) * count
        mse = squared_diff.sum() / N  # global normalization
        rms_error_i = np.sqrt(mse)

        class_errors.append(rms_error_i)

    cw_ada_sce_rms = np.mean(class_errors) if class_errors else 0.0
    return cw_ada_sce_rms

def compute_cc_adasce_rms(conf_class, true, num_classes, conf_bin_num=10, threshold=0.01):
    """
    RMS Class-Conditional Adaptive Static Calibration Error (CC-AdaSCE-RMS).

    Computes the root mean squared calibration error conditioned on the true class.

    Args:
        conf_class (np.ndarray): Shape (N, C), predicted class probabilities.
        true (np.ndarray): Shape (N,), ground truth labels.
        num_classes (int): Total number of classes.
        conf_bin_num (int): Number of quantile bins (default: 10).
        threshold (float): Minimum confidence to include in calibration (default: 0.01).

    Returns:
        float: CC-AdaSCE-RMS value.
    """
    N = len(true)
    class_errors = []

    for k in range(num_classes):
        p_k = conf_class[:, k]
        y_k = (true == k).astype(int)

        df = pd.DataFrame({'p_k': p_k, 'y_k': y_k})
        df = df[df['y_k'] == 1]  # Only samples where the true label is class k
        df = df[df['p_k'] >= threshold]

        if len(df) < 2:
            continue

        try:
            df['bin'] = pd.qcut(df['p_k'], q=conf_bin_num, duplicates='drop', labels=False)
        except ValueError:
            continue  # Skip if not enough unique values

        group_conf = df.groupby('bin')['p_k'].mean()
        group_acc = df.groupby('bin')['y_k'].mean()  # will be 1.0 for all
        bin_counts = df.groupby('bin')['p_k'].count()

        valid_idx = group_acc.index.intersection(group_conf.index)
        acc = group_acc.loc[valid_idx]
        conf = group_conf.loc[valid_idx]
        count = bin_counts.loc[valid_idx]

        squared_diff = ((acc - conf) ** 2) * count
        mse = squared_diff.sum() / N  # global normalization
        rms_error_k = np.sqrt(mse)

        class_errors.append(rms_error_k)

    cc_adasce_rms = np.mean(class_errors) if class_errors else 0.0
    return cc_adasce_rms


def compute_sharpness(conf_class):
    """
    Sharpness (average entropy)

    Sharpness measures the concentration of the predictive distribution.
    Lower entropy indicates more confident predictions (sharper),
    while higher entropy indicates more uncertain (smoother) predictions.

    Args:
        conf_class (np.ndarray): List of confidences for all classes

    Returns:
        sharpness (float): Sharpness value. Lower is sharper.
    """
    
    probs = np.clip(conf_class, 1e-12, 1.0)
    entropy = -np.sum(probs * np.log(probs), axis=1)
    sharpness = np.mean(entropy)
    return sharpness

def compute_nll(conf_class, true, num_classes):
    """
    Negative Log-Likelihood (NLL) of the predicted class probabilities.

    NLL is a proper scoring rule that penalizes low confidence in the true label.

    Args:
        conf_class (np.ndarray): List of confidences for all classes
        true (np.ndarray): List of true labels
        num_classes (int): Number of classes

    Returns:
        nll (float): NLL value. Lower values indicate better accuracy and confidence in correct predictions.
    """
    
    cce = tf.keras.losses.CategoricalCrossentropy()
    true = tf.one_hot(true, depth=num_classes)
    nll = cce(true, conf_class).numpy()
    return nll

def compute_brier_score(conf_class, true, num_classes):
    """
    Brier score

    The Brier score measures the mean squared difference between predicted
    probabilities and the one-hot encoded true labels.

    Args:
        conf_class (np.ndarray): List of confidences for all classes
        true (np.ndarray): List of true labels
        num_classes (int): Total number of classes

    Returns:
        brier_score (float): Brier score value. Lower values indicate better calibration and accuracy.
    """
    
    true = tf.one_hot(true, depth=num_classes)
    brier_score = np.mean(np.sum((conf_class - true)**2, axis=1))
    return brier_score
    
#-----------------------------------
# Other Metrics
#-----------------------------------
    
def compute_classification_error(metric, true_y, k=1):
    count = 0
    for i in range(len(metric)):
        idx = np.argsort(metric[i])[-k:]
        if true_y[i] in idx:
            count += 1
    acc = count / len(metric)
    return 1 - acc