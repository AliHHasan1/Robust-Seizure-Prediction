import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix

def calc_metrics(y, preds):
    auc_score = roc_auc_score(y, preds)
    preds = np.argmax(preds, axis=1)
    y = np.argmax(y, axis=1)
    c = confusion_matrix(y, preds, labels=[0, 1])
    sens = c[1][1] / (c[1][1] + c[1][0]) if (c[1][1] + c[1][0]) > 0 else 0
    fpr = c[0][1] / (c[0][1] + c[0][0]) if (c[0][1] + c[0][0]) > 0 else 0
    acc = (c[1][1] + c[0][0]) / np.sum(c) if np.sum(c) > 0 else 0
    return auc_score, acc, sens, fpr

def shuffle_data(X, y):
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    return X[idx], y[idx]

def next_batch(X, y, counter, batch_size):
    start_index = counter * batch_size
    end_index = start_index + batch_size
    if end_index > X.shape[0]:
        end_index = X.shape[0]
    return X[start_index:end_index], y[start_index:end_index]

def train_val_cv_split(ictal_X, ictal_y, interictal_X, interictal_y, val_ratio=0.1, is_shuffeling=True, is_shuffling=None):
    if is_shuffling is not None:
        is_shuffeling = bool(is_shuffling)

    nfold = min(len(ictal_y), len(interictal_y))
    print('number of folds= ', nfold)

    for i in range(nfold):
        X_test_ictal = ictal_X[i]
        y_test_ictal = ictal_y[i]
        X_test_interictal = interictal_X[i]
        y_test_interictal = interictal_y[i]

        if i == 0:
            X_train_ictal = np.concatenate(ictal_X[1:], axis=0)
            y_train_ictal = np.concatenate(ictal_y[1:], axis=0)
            X_train_interictal = np.concatenate(interictal_X[1:], axis=0)
            y_train_interictal = np.concatenate(interictal_y[1:], axis=0)
        elif i < nfold - 1:
            # Robust concat across folds even when folds have different number of windows.
            # (interictal folds are often uneven due to splitting.)
            X_train_ictal = np.concatenate(ictal_X[:i] + ictal_X[i + 1:], axis=0)
            y_train_ictal = np.concatenate(ictal_y[:i] + ictal_y[i + 1:], axis=0)
            X_train_interictal = np.concatenate(interictal_X[:i] + interictal_X[i + 1:], axis=0)
            y_train_interictal = np.concatenate(interictal_y[:i] + interictal_y[i + 1:], axis=0)
        else:
            X_train_ictal = np.concatenate(ictal_X[:i], axis=0)
            y_train_ictal = np.concatenate(ictal_y[:i], axis=0)
            X_train_interictal = np.concatenate(interictal_X[:i], axis=0)
            y_train_interictal = np.concatenate(interictal_y[:i], axis=0)

        X_train_interictal = X_train_interictal[0:X_train_ictal.shape[0]]
        y_train_interictal = y_train_interictal[0:y_train_ictal.shape[0]]
        X_test_interictal = X_test_interictal[0:X_test_ictal.shape[0]]
        y_test_interictal = y_test_interictal[0:y_test_ictal.shape[0]]

        X_train = np.concatenate(
            (
                X_train_ictal[:int(X_train_ictal.shape[0] * (1 - val_ratio))],
                X_train_interictal[:int(X_train_interictal.shape[0] * (1 - val_ratio))]
            ),
            axis=0
        )
        y_train = np.concatenate(
            (
                y_train_ictal[:int(X_train_ictal.shape[0] * (1 - val_ratio))],
                y_train_interictal[:int(X_train_interictal.shape[0] * (1 - val_ratio))]
            ),
            axis=0
        )
        if is_shuffeling:
            X_train, y_train = shuffle_data(X_train, y_train)

        X_val = np.concatenate(
            (
                X_train_ictal[int(X_train_ictal.shape[0] * (1 - val_ratio)):],
                X_train_interictal[int(X_train_interictal.shape[0] * (1 - val_ratio)):]
            ),
            axis=0
        )
        y_val = np.concatenate(
            (
                y_train_ictal[int(X_train_ictal.shape[0] * (1 - val_ratio)):],
                y_train_interictal[int(X_train_interictal.shape[0] * (1 - val_ratio)):]
            ),
            axis=0
        )

        X_test = np.concatenate((X_test_ictal, X_test_interictal), axis=0)
        y_test = np.concatenate((y_test_ictal, y_test_interictal), axis=0)
        yield (X_train, y_train, X_val, y_val, X_test, y_test)

def train_val_test_split(ictal_X, ictal_y, interictal_X, interictal_y, val_ratio=0.1, test_ratio=0.2, is_shuffeling=True):
    num_sz = len(ictal_y)
    num_sz_test = int(np.ceil(test_ratio * num_sz))
    print('Total %d seizures. Last %d is used for testing.' % (num_sz, num_sz_test))

    interictal_X = interictal_X[0: len(ictal_y)]
    interictal_y = interictal_y[0: len(ictal_y)]
    interictal_X = np.concatenate(interictal_X, axis=0)
    interictal_y = np.concatenate(interictal_y, axis=0)

    X_test_ictal = np.concatenate(ictal_X[-num_sz_test:], axis=0)
    y_test_ictal = np.concatenate(ictal_y[-num_sz_test:], axis=0)
    X_test_interictal = interictal_X[-num_sz_test:]
    y_test_interictal = interictal_y[-num_sz_test:]

    X_train_ictal = np.concatenate(ictal_X[:-num_sz_test], axis=0)
    y_train_ictal = np.concatenate(ictal_y[:-num_sz_test], axis=0)
    X_train_interictal = interictal_X[:-num_sz_test]
    y_train_interictal = interictal_y[:-num_sz_test]

    down_spl = int(np.floor(y_train_interictal.shape[0] / y_train_ictal.shape[0]))
    if down_spl > 1:
        X_train_interictal = X_train_interictal[::down_spl]
        y_train_interictal = y_train_interictal[::down_spl]
    elif down_spl == 1:
        X_train_interictal = X_train_interictal[:X_train_ictal.shape[0]]
        y_train_interictal = y_train_interictal[:X_train_ictal.shape[0]]

    X_train = np.concatenate(
        (
            X_train_ictal[:int(X_train_ictal.shape[0] * (1 - val_ratio))],
            X_train_interictal[:int(X_train_interictal.shape[0] * (1 - val_ratio))]
        ),
        axis=0
    )
    y_train = np.concatenate(
        (
            y_train_ictal[:int(X_train_ictal.shape[0] * (1 - val_ratio))],
            y_train_interictal[:int(X_train_interictal.shape[0] * (1 - val_ratio))]
        ),
        axis=0
    )
    if is_shuffeling:
        X_train, y_train = shuffle_data(X_train, y_train)

    X_val = np.concatenate(
        (
            X_train_ictal[int(X_train_ictal.shape[0] * (1 - val_ratio)):],
            X_train_interictal[int(X_train_interictal.shape[0] * (1 - val_ratio)):]
        ),
        axis=0
    )
    y_val = np.concatenate(
        (
            y_train_ictal[int(X_train_ictal.shape[0] * (1 - val_ratio)):],
            y_train_interictal[int(X_train_interictal.shape[0] * (1 - val_ratio)):]
        ),
        axis=0
    )

    X_test = np.concatenate((X_test_ictal, X_test_interictal), axis=0)
    y_test = np.concatenate((y_test_ictal, y_test_interictal), axis=0)
    return X_train, y_train, X_val, y_val, X_test, y_test

def collect_results(history, sensitivity, false_alarm, preds, y_test, test_loss1, emb):
    history['test_sensitivity'].append(sensitivity)
    history['test_false_alarm'].append(false_alarm)
    history['y_pred'].append(preds)
    history['y_test'].append(y_test)
    history['test_loss'].append(test_loss1)
    history['embed1'].append(emb)
    return history
