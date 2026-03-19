import os
import numpy as np
import pickle

def save_hickle_file(filename, data):
    """حفظ المصفوفات (X, y) باستخدام numpy لضمان السرعة والتوافق"""
    path = filename + '.npy'
    print(f'Saving data to {path}')
    np.save(path, data)

def load_hickle_file(filename):
    """تحميل المصفوفات"""
    path = filename + '.npy'
    if os.path.isfile(path):
        print(f'Loading {path} ...')
        return np.load(path, allow_pickle=True)
    return None

def save_history(history, file_path):
    """حفظ سجل التدريب (Accuracy/Loss) بصيغة pickle"""
    with open(file_path + '.pkl', 'wb') as f:
        pickle.dump(history, f)