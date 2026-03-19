import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, auc, roc_curve, confusion_matrix

def load_results(path, dataset, patients_list=None):
    """
    تحميل النتائج للمرضى من المجلد المحدد.
    إذا لم يتم تمرير قائمة مرضى، سيستخدم القائمة الافتراضية لـ CHBMIT.
    """
    if patients_list is None:
        if dataset == 'CHBMIT':
            patients = ['1', '2', '3', '5', '9', '10', '13', '14', '18', '19', '20', '21', '23']
        else:
            patients = ['1', '3', '4', '5', '6', '14', '15', '16', '17', '18', '19', '20', '21']
    else:
        patients = patients_list

    data = dict()
    for i in patients:
        # البحث عن ملف pkl أو npy حسب نظام الحفظ الجديد
        file_path = os.path.join(path, f'history_{i}.pkl')
        if os.path.isfile(file_path):
            with open(file_path, 'rb') as f:
                data[i] = pickle.load(f)
        else:
            print(f"Warning: Result file for patient {i} not found at {file_path}")
    
    return data, patients

def calculate_fpr(pred, test):
    """
    حساب معدل الإنذارات الكاذبة (FPR/h) بناءً على معيار البحث الأصلي:
    - تقسيم البيانات لفترات 5 دقائق.
    - إطلاق إنذار إذا كانت 8 نوافذ من أصل 10 إيجابية.
    - الإنذار الواحد يغطي فترة 35 دقيقة.
    """
    fpr = 0
    seg = []
    
    # نأخذ التوقعات الخاصة بفترات ما بين النوبات فقط (Interictal) لحساب الإنذارات الكاذبة
    interictal_indices = np.where(test == 0)[0]
    if len(interictal_indices) == 0:
        return 0
        
    pred_interictal = pred[interictal_indices]
    
    # تحويل النوافذ (كل نافذة تمثل جزء من الـ 30 ثانية الأصلية في البحث) إلى قطع 5 دقائق
    n_five_mins = (len(pred_interictal) // 10) + 1
    
    counter = 0
    while (counter + 1) <= n_five_mins:
        win = pred_interictal[counter*10 : (counter+1)*10]
        if len(win) > 0:
            # إذا كان هناك 8 نوافذ إيجابية أو أكثر من أصل 10
            r = np.count_nonzero(win)
            seg.append(1 if r >= 8 else 0)
        counter += 1
    
    # منطق استمرار الإنذار لمدة 35 دقيقة (7 قطع من فئة 5 دقائق)
    j = 0
    if len(seg) > 0:
        while j + 7 <= len(seg):
            if seg[j] == 1:
                fpr += 1  # تسجيل إنذار كاذب واحد
                j += 7    # قفز 35 دقيقة لأن الإنذار مستمر
            else:
                j += 1
        
        # حساب المعدل لكل ساعة (Total Hours = Total 5-min segments * 5 / 60)
        total_hours = (n_five_mins * 5) / 60
        fpr_per_hour = fpr / total_hours if total_hours > 0 else 0
    else:
        fpr_per_hour = 0
        
    return fpr_per_hour

def summary_results(patients, data):
    """
    طباعة ملخص شامل للنتائج لكل مريض والمتوسط العام.
    """
    all_sens = []
    all_fpr = []
    
    print(f"\n{'Patient':<10} | {'Sensitivity (%)':<15} | {'FPR/h':<10}")
    print("-" * 45)
    
    for i in patients:
        if i not in data: continue
        
        # استخراج الحساسية من سجل التدريب
        sen_mean = np.mean(data[i].get('test_sensitivity', [0])) * 100
        
        # حساب FPR باستخدام الدالة الطبية
        y_pred = np.array(data[i].get('y_pred', []))
        y_test = np.array(data[i].get('y_test', []))
        
        if len(y_pred) > 0 and len(y_test) > 0:
            # تحويل من One-hot encoding إلى Class ID إذا لزم الأمر
            if y_pred.ndim > 1: y_pred = np.argmax(y_pred, axis=-1)
            if y_test.ndim > 1: y_test = np.argmax(y_test, axis=-1)
            
            fpr = calculate_fpr(y_pred, y_test)
        else:
            fpr = 0
            
        all_sens.append(sen_mean)
        all_fpr.append(fpr)
        
        print(f"Patient {i:<2} | {sen_mean:<15.2f} | {fpr:<10.4f}")
    
    if all_sens:
        print("-" * 45)
        print(f"AVERAGE    | {np.mean(all_sens):<15.2f} | {np.mean(all_fpr):<10.4f}")

def auc_results(data, patients):
    """حساب متوسط الـ AUC للمرضى"""
    results = []
    for i in patients:
        if i in data and 'AUC_AVG' in data[i]:
            results.append(data[i]['AUC_AVG'][0])
    return np.array(results)