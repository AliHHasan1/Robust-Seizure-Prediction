import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, auc, roc_curve, confusion_matrix

def load_results(path, dataset, patients_list=None):
    """
    تحميل النتائج للمرضى من المجلد المحدد.
    يدعم تنسيق pkl (الجديد) و hkl (القديم إذا لزم الأمر).
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
        # البحث عن ملف pkl (التنسيق المعتمد في TF2)
        file_path = os.path.join(path, f'history_{i}.pkl')
        if os.path.isfile(file_path):
            with open(file_path, 'rb') as f:
                data[i] = pickle.load(f)
        else:
            # محاولة البحث عن التنسيق القديم history1.hkl إذا كان موجوداً
            old_file_path = os.path.join(path, f'history{i}.hkl')
            if os.path.isfile(old_file_path):
                try:
                    import hickle as hkl
                    data[i] = hkl.load(old_file_path)
                except ImportError:
                    print(f"Warning: Found .hkl file but 'hickle' library is not installed.")
            else:
                print(f"Warning: Result file for patient {i} not found.")
    
    return data, patients

def calculate_fpr(pred, test):
    """
    حساب معدل الإنذارات الكاذبة (FPR/h) بناءً على المنهجية العلمية الأصلية:
    - يتم حساب الإنذارات الكاذبة فقط في فترات ما بين النوبات (Interictal).
    - كل 10 نوافذ (كل نافذة 28-30 ثانية) تشكل قطعة زمنية مدتها 5 دقائق.
    - يطلق إنذار إذا كانت 8 نوافذ من أصل 10 إيجابية (Seizure Predicted).
    - الإنذار الواحد يغطي فترة 35 دقيقة (أي يتم تجاهل أي إنذارات أخرى خلال هذه الفترة).
    """
    fpr_count = 0
    seg = []
    
    # 1. استخراج التوقعات الخاصة بفترات Interictal فقط (Label = 0)
    # ملاحظة: التوقعات يجب أن تكون Class IDs (0 or 1)
    interictal_indices = np.where(test == 0)[0]
    if len(interictal_indices) == 0:
        return 0
        
    pred_interictal = pred[interictal_indices]
    
    # 2. تحويل النوافذ إلى قطع 5 دقائق (كل قطعة تحتوي 10 نوافذ)
    # المنهجية الأصلية: n_five_mins = (len(pred)/10) + 1
    n_five_mins = (len(pred_interictal) // 10)
    
    for i in range(n_five_mins):
        win = pred_interictal[i*10 : (i+1)*10]
        # إذا كان هناك 8 نوافذ إيجابية أو أكثر من أصل 10
        r = np.count_nonzero(win)
        seg.append(1 if r >= 8 else 0)
    
    # 3. منطق استمرار الإنذار لمدة 35 دقيقة (7 قطع من فئة 5 دقائق)
    j = 0
    while j < len(seg):
        if seg[j] == 1:
            fpr_count += 1  # تسجيل إنذار كاذب
            j += 7    # قفز 35 دقيقة (7 قطع * 5 دقائق) لأن الإنذار مستمر
        else:
            j += 1
    
    # 4. حساب المعدل لكل ساعة
    # إجمالي الساعات = (عدد قطع الـ 5 دقائق * 5) / 60
    total_hours = (n_five_mins * 5) / 60
    fpr_per_hour = fpr_count / total_hours if total_hours > 0 else 0
    
    return fpr_per_hour

def calculate_sensitivity(y_true, y_pred):
    """
    حساب الحساسية (Sensitivity/Recall) بدقة من مصفوفة الارتباك.
    """
    if len(y_true) == 0 or len(y_pred) == 0:
        return 0
    
    # التأكد من أنها Class IDs وليست One-hot
    if y_true.ndim > 1: y_true = np.argmax(y_true, axis=-1)
    if y_pred.ndim > 1: y_pred = np.argmax(y_pred, axis=-1)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    return sensitivity * 100

def summary_results(patients, data):
    """
    طباعة ملخص شامل للنتائج لكل مريض والمتوسط العام.
    """
    all_sens = []
    all_fpr = []
    all_auc = []
    
    print(f"\n{'Patient':<10} | {'Sensitivity (%)':<15} | {'FPR/h':<10} | {'AUC':<10}")
    print("-" * 55)
    
    for i in patients:
        if i not in data: continue
        
        # 1. استخراج البيانات
        y_pred_raw = np.array(data[i].get('y_pred', []))
        y_test_raw = np.array(data[i].get('y_test', []))
        
        if len(y_pred_raw) == 0 or len(y_test_raw) == 0:
            continue

        # 2. تحويل البيانات للمقاييس
        # للـ AUC نحتاج الاحتمالات (Probabilities)
        # للـ Sensitivity و FPR نحتاج Class IDs
        y_test_ids = np.argmax(y_test_raw, axis=-1) if y_test_raw.ndim > 1 else y_test_raw
        y_pred_ids = np.argmax(y_pred_raw, axis=-1) if y_pred_raw.ndim > 1 else y_pred_raw
        
        # 3. حساب المقاييس
        sens = calculate_sensitivity(y_test_ids, y_pred_ids)
        fpr = calculate_fpr(y_pred_ids, y_test_ids)
        
        # حساب AUC (باستخدام احتمالات الفئة 1)
        try:
            if y_pred_raw.ndim > 1:
                auc_val = roc_auc_score(y_test_ids, y_pred_raw[:, 1])
            else:
                auc_val = roc_auc_score(y_test_ids, y_pred_raw)
        except:
            auc_val = data[i].get('AUC_AVG', [0])[0]

        all_sens.append(sens)
        all_fpr.append(fpr)
        all_auc.append(auc_val)
        
        print(f"Patient {i:<2} | {sens:<15.2f} | {fpr:<10.4f} | {auc_val:<10.4f}")
    
    if all_sens:
        print("-" * 55)
        print(f"AVERAGE    | {np.mean(all_sens):<15.2f} | {np.mean(all_fpr):<10.4f} | {np.mean(all_auc):<10.4f}")

def generate_final_report(patients, data, output_dir="results/final_report"):
    """
    توليد تقرير نهائي مجمع لجميع المرضى مع رسوم بيانية احترافية.
    هذه الدالة تحاكي منهجية العرض في المستودع الأصلي وتضيف عليها لمسة احترافية.
    """
    os.makedirs(output_dir, exist_ok=True)
    report_data = []
    
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    for i in patients:
        if i not in data: continue
        
        y_pred_raw = np.array(data[i].get('y_pred', []))
        y_test_raw = np.array(data[i].get('y_test', []))
        
        if len(y_pred_raw) == 0 or len(y_test_raw) == 0: continue
        
        y_test_ids = np.argmax(y_test_raw, axis=-1) if y_test_raw.ndim > 1 else y_test_raw
        y_pred_probs = y_pred_raw[:, 1] if y_pred_raw.ndim > 1 else y_pred_raw
        y_pred_ids = np.argmax(y_pred_raw, axis=-1) if y_pred_raw.ndim > 1 else y_pred_raw
        
        # حساب المقاييس
        sens = calculate_sensitivity(y_test_ids, y_pred_ids)
        fpr_h = calculate_fpr(y_pred_ids, y_test_ids)
        auc_val = roc_auc_score(y_test_ids, y_pred_probs)
        
        report_data.append({
            'Patient': i,
            'Sensitivity (%)': round(sens, 2),
            'FPR/h': round(fpr_h, 4),
            'AUC': round(auc_val, 4)
        })
        
        # رسم منحنى ROC لكل مريض
        fpr, tpr, _ = roc_curve(y_test_ids, y_pred_probs)
        plt.plot(fpr, tpr, label=f'Patient {i} (AUC = {auc_val:.2f})', alpha=0.6)
    
    # إعدادات الرسم البياني المجمع
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Combined ROC Curves for All Patients')
    plt.legend(loc="lower right", fontsize='small', ncol=2)
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'combined_roc_curves.png'))
    plt.close()
    
    # حفظ التقرير في ملف CSV
    df = pd.DataFrame(report_data)
    df.to_csv(os.path.join(output_dir, 'final_results_summary.csv'), index=False)
    
    # إضافة سطر المتوسط العام
    avg_row = {
        'Patient': 'AVERAGE',
        'Sensitivity (%)': round(df['Sensitivity (%)'].mean(), 2),
        'FPR/h': round(df['FPR/h'].mean(), 4),
        'AUC': round(df['AUC'].mean(), 4)
    }
    df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)
    
    print(f"\nFinal Report generated in: {output_dir}")
    print(df.to_string(index=False))
    
    return df