import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix

def calc_metrics(y_true, y_pred_probs):
    """
    حساب المقاييس الأساسية (AUC, Accuracy, Sensitivity, FPR).
    y_true: One-hot labels
    y_pred_probs: Probabilities from softmax
    """
    # حساب AUC باستخدام الاحتمالات
    auc_score = roc_auc_score(y_true, y_pred_probs)
    
    # تحويل إلى Class IDs للمقاييس الأخرى
    y_pred_ids = np.argmax(y_pred_probs, axis=1)
    y_true_ids = np.argmax(y_true, axis=1)
    
    c = confusion_matrix(y_true_ids, y_pred_ids, labels=[0, 1])
    
    # الحساسية (Sensitivity) = TP / (TP + FN)
    sens = c[1, 1] / (c[1, 1] + c[1, 0]) if (c[1, 1] + c[1, 0]) > 0 else 0
    
    # معدل الإيجابيات الكاذبة (FPR) = FP / (FP + TN)
    fpr = c[0, 1] / (c[0, 1] + c[0, 0]) if (c[0, 1] + c[0, 0]) > 0 else 0
    
    # الدقة (Accuracy)
    acc = (c[1, 1] + c[0, 0]) / np.sum(c) if np.sum(c) > 0 else 0
    
    return auc_score, acc, sens, fpr

def shuffle_data(X, y):
    """خلط البيانات عشوائياً"""
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    return X[idx], y[idx]

def train_val_cv_split(ictal_X, ictal_y, interictal_X, interictal_y, val_ratio=0.1, is_shuffling=True):
    """
    منهجية Leave-One-Seizure-Out Cross-Validation.
    في كل Fold، يتم استبعاد نوبة كاملة للاختبار، والباقي للتدريب.
    هذا يضمن عدم حدوث تسريب للبيانات (Data Leakage).
    """
    # عدد الـ Folds يساوي عدد النوبات المتاحة
    n_folds = len(ictal_X)
    print(f'Number of folds (Seizures): {n_folds}')

    for i in range(n_folds):
        # 1. تحديد بيانات الاختبار (Fold الحالي)
        X_test_ictal = ictal_X[i]
        y_test_ictal = ictal_y[i]
        
        # موازنة بيانات الاختبار (Interictal) لتناسب حجم النوبة المستبعدة
        X_test_interictal = interictal_X[i][:len(X_test_ictal)]
        y_test_interictal = interictal_y[i][:len(y_test_ictal)]
        
        # 2. تجميع بيانات التدريب (باقي الـ Folds)
        X_train_ictal_list = [ictal_X[j] for j in range(n_folds) if j != i]
        y_train_ictal_list = [ictal_y[j] for j in range(n_folds) if j != i]
        
        X_train_interictal_list = [interictal_X[j] for j in range(n_folds) if j != i]
        y_train_interictal_list = [interictal_y[j] for j in range(n_folds) if j != i]
        
        X_train_ictal = np.concatenate(X_train_ictal_list, axis=0)
        y_train_ictal = np.concatenate(y_train_ictal_list, axis=0)
        
        X_train_interictal = np.concatenate(X_train_interictal_list, axis=0)
        y_train_interictal = np.concatenate(y_train_interictal_list, axis=0)
        
        # 3. موازنة بيانات التدريب (Downsampling Interictal)
        X_train_interictal = X_train_interictal[:len(X_train_ictal)]
        y_train_interictal = y_train_interictal[:len(y_train_ictal)]
        
        # 4. تقسيم التدريب إلى (تدريب + تحقق)
        split_idx = int(len(X_train_ictal) * (1 - val_ratio))
        
        X_train = np.concatenate([X_train_ictal[:split_idx], X_train_interictal[:split_idx]], axis=0)
        y_train = np.concatenate([y_train_ictal[:split_idx], y_train_interictal[:split_idx]], axis=0)
        
        X_val = np.concatenate([X_train_ictal[split_idx:], X_train_interictal[split_idx:]], axis=0)
        y_val = np.concatenate([y_train_ictal[split_idx:], y_train_interictal[split_idx:]], axis=0)
        
        X_test = np.concatenate([X_test_ictal, X_test_interictal], axis=0)
        y_test = np.concatenate([y_test_ictal, y_test_interictal], axis=0)
        
        if is_shuffling:
            X_train, y_train = shuffle_data(X_train, y_train)
            X_val, y_val = shuffle_data(X_val, y_val)
            
        yield (X_train, y_train, X_val, y_val, X_test, y_test)

def collect_results(history, metrics, preds, y_test):
    """تجميع النتائج لكل Fold"""
    auc_val, acc, sens, fpr = metrics
    history['test_sensitivity'].append(sens)
    history['test_fpr'].append(fpr)
    history['test_accuracy'].append(acc)
    history['AUC_AVG'].append(auc_val)
    history['y_pred'].append(preds)
    history['y_test'].append(y_test)
    return history
