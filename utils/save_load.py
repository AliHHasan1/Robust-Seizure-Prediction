import os
import numpy as np
import pickle

def save_hickle_file(filename, data):
    """
    حفظ المصفوفات (X, y) باستخدام numpy لضمان السرعة والتوافق مع TensorFlow 2.
    تم الحفاظ على اسم الدالة الأصلي لضمان عدم كسر الاستدعاءات في الأجزاء الأخرى من الكود.
    """
    # التأكد من أن المسار ينتهي بـ .npy
    if not filename.endswith('.npy'):
        path = filename + '.npy'
    else:
        path = filename
        
    print(f'Saving data to {path}')
    # استخدام np.save للحفظ السريع للمصفوفات الضخمة
    np.save(path, data)

def load_hickle_file(filename):
    """
    تحميل المصفوفات مع دعم التنسيق الجديد (.npy) والقديم (.hkl) إذا لزم الأمر.
    """
    # 1. محاولة التحميل بتنسيق numpy الجديد
    npy_path = filename if filename.endswith('.npy') else filename + '.npy'
    if os.path.isfile(npy_path):
        print(f'Loading {npy_path} ...')
        return np.load(npy_path, allow_pickle=True)
    
    # 2. محاولة التحميل بتنسيق hickle القديم (للتوافق مع البيانات السابقة)
    hkl_path = filename if filename.endswith('.hkl') else filename + '.hkl'
    if os.path.isfile(hkl_path):
        try:
            import hickle as hkl
            print(f'Loading legacy file {hkl_path} ...')
            return hkl.load(hkl_path)
        except ImportError:
            print(f"Warning: Found legacy .hkl file but 'hickle' library is not installed.")
            
    return None

def savefile(data, file_path):
    """
    دالة عامة للحفظ (تستخدم لحفظ سجلات التدريب أو أي بيانات أخرى).
    تستخدم تنسيق pickle لضمان التوافق مع TF2.
    """
    # التأكد من وجود المجلد
    os.makedirs(os.path.dirname(file_path), exist_ok=True) if os.path.dirname(file_path) else None
    
    # إضافة الامتداد إذا لم يكن موجوداً
    if not file_path.endswith('.pkl'):
        path = file_path + '.pkl'
    else:
        path = file_path
        
    print(f'Saving to {path}')
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def load_ae(path, target):
    """
    تحميل الأمثلة العدائية (Adversarial Examples) المولدة مسبقاً.
    تدعم التنسيقات الجديدة والقديمة.
    """
    # تحديد أسماء الملفات المتوقعة
    input_base = os.path.join(path, f'augmented_input{target}')
    labels_base = os.path.join(path, f'augmented_lables{target}')
    
    # محاولة تحميل البيانات
    data = load_hickle_file(input_base)
    labels = load_hickle_file(labels_base)
    
    if data is not None and labels is not None:
        return data, labels
    
    return None, None
