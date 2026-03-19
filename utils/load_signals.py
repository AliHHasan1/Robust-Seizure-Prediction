import os
import numpy as np
import pandas as pd
import mne
from mne.io import read_raw_edf
from sklearn.preprocessing import StandardScaler

def get_channels_by_subject(subject_id):
    """
    التعريف العلمي للقنوات حسب المريض (تخصيص Spatial Coverage).
    """
    subject_id = str(subject_id)
    base_chs = [
        'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 
        'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2', 
        'FZ-CZ', 'CZ-PZ'
    ]
    extra_chs = ['P7-T7', 'T7-FT9', 'FT9-FT10', 'FT10-T8']

    if subject_id in ['13', '16']:
        return base_chs[:-1] 
    elif subject_id == '4':
        special_chs = base_chs.copy()
        special_chs.remove('T8-P8')
        return special_chs + ['P8-O2', 'FZ-CZ', 'CZ-PZ', 'P7-T7', 'T7-FT9', 'FT10-T8']
    else:
        return base_chs + extra_chs

def get_previous_file_name(current_file):
    """استنتاج اسم الملف السابق لربط الإشارات متصلة زمنياً"""
    try:
        prefix = current_file.split('_')[0] 
        seq_str = current_file.split('_')[1].split('.')[0]
        seq = int(seq_str)
        return f"{prefix}_{seq-1:02d}.edf"
    except:
        return None

def load_preictal_segment(data_dir, patient_id, edf_file, sz_start_sec):
    """
    اقتطاع 30 دقيقة (SOP) قبل النوبة بـ 5 دقائق (SPH).
    """
    fs = 256 
    SOP = 30 * 60 * fs 
    SPH = 5 * 60 * fs  
    
    st = int(sz_start_sec * fs) - SPH - SOP
    sp = int(sz_start_sec * fs) - SPH
    
    channels = get_channels_by_subject(patient_id)
    current_file_path = os.path.join(data_dir, f"chb{int(patient_id):02d}", edf_file)
    
    raw = read_raw_edf(current_file_path, preload=True, verbose=False)
    raw.pick_channels(channels)
    raw.notch_filter(np.arange(60, 121, 60), fir_design='firwin', verbose=False)
    current_data = raw.get_data().T 
    
    if st < 0:
        prev_file = get_previous_file_name(edf_file)
        
        if prev_file:
            prev_path = os.path.join(data_dir, f"chb{int(patient_id):02d}", prev_file)
            
            if os.path.exists(prev_path):
                prev_raw = read_raw_edf(prev_path, preload=True, verbose=False)
                prev_raw.pick_channels(channels)
                prev_raw.notch_filter(np.arange(60, 121, 60), fir_design='firwin', verbose=False)
                
                prev_data = prev_raw.get_data().T
                
                data = np.concatenate((prev_data[st:], current_data[:sp]), axis=0)
                
                # ✅ تحقق من طول SOP
                if data.shape[0] != SOP:
                    return None
                
                return data
        
        # fallback إذا ما في ملف سابق
        data = current_data[0:sp]
        
        if data.shape[0] != SOP:
            return None
        
        return data

    else:
        data = current_data[st:sp]
        
        # ✅ تحقق من طول SOP
        if data.shape[0] != SOP:
            return None
        
        return data


def load_special_interictal_metadata(metadata_dir):
    """
    تحميل ملف الاستثناءات للحالات الطبيعية.
    علمياً: بعض التسجيلات الطويلة تحتوي على فترات غير صالحة للتحليل، 
    هذا الملف يحدد الأجزاء 'النظيفة' حصراً.
    """
    path = os.path.join(metadata_dir, 'special_interictal.csv')
    if os.path.exists(path):
        # قراءة الملف بدون رؤوس أعمدة وتسميتها يدوياً للدقة
        return pd.read_csv(path, header=None, names=['File_name', 'Start', 'End'])
    return pd.DataFrame(columns=['File_name', 'Start', 'End'])

def load_interictal_segment(data_dir, metadata_dir, patient_id, edf_file):
    """
    تحميل الحالة الطبيعية (Interictal) للمريض.
    يتم هنا تطبيق عمليات القص (Cropping) بناءً على القيود الزمنية في special_interictal.
    """
    channels = get_channels_by_subject(patient_id)
    file_path = os.path.join(data_dir, f"chb{int(patient_id):02d}", edf_file)
    
    # 1. تحميل الإشارة الخام
    raw = read_raw_edf(file_path, preload=True, verbose=False)
    raw.pick_channels(channels)
    
    # 2. فحص وجود استثناءات زمنية لهذا الملف
    special_df = load_special_interictal_metadata(metadata_dir)
    file_constraints = special_df[special_df['File_name'] == edf_file]
    
    if not file_constraints.empty:
        # إذا وجد قيد زمني، نقوم بقص الإشارة (مثلاً من الثانية X إلى الثانية Y)
        t_start = float(file_constraints.iloc[0]['Start'])
        t_end = float(file_constraints.iloc[0]['End'])
        
        # إذا كانت القيمة -1 تعني حتى نهاية الملف
        t_end = None if t_end == -1 else t_end
        raw.crop(tmin=t_start, tmax=t_end)
    raw.notch_filter(np.arange(60, 121, 60), fir_design='firwin', verbose=False)
    data = raw.get_data().T
    if data.shape[0] == 0:
        return None
    return data

  

def apply_scaling(data):
    """
    تطبيق التطبيع الإحصائي (Standardization).
    هندسياً: الحفاظ على الخصائص الديناميكية للإشارة مع توحيد النطاق الرقمي 
    لجميع القنوات لتسريع تقارب (Convergence) النموذج أثناء التدريب.
    """
    scaler = StandardScaler()
    # يتم حساب المتوسط والانحراف المعياري لكل قناة على حدة
    scaled_data = scaler.fit_transform(data)
    return scaled_data

def create_windows(data, label_value):
    """
    تقطيع الإشارة إلى نوافذ زمنية ثابتة الطول.
    المدخلات: المصفوفة الموحدة (data)، والقيمة التصنيفية (label_value: 0 or 1).
    """
    fs = 256
    window_sec = 28
    overlap_percent = 0.5
    
    window_len = window_sec * fs  # 7168 عينة
    step_len = int(window_len * (1 - overlap_percent)) # القفزة: 14 ثانية (3584 عينة)
    
    windows_X = []
    windows_y = []
    
    # حلقة التقطيع (Sliding Window Loop)
    start_idx = 0
    while start_idx + window_len <= len(data):
        window = data[start_idx : start_idx + window_len, :]
        windows_X.append(window)
        windows_y.append(label_value)
        
        # الانتقال للنافذة التالية بناءً على التداخل
        start_idx += step_len
        
    return np.array(windows_X, dtype='float32'), np.array(windows_y, dtype='float32')

def prepare_dataset_by_mode(data_dir, metadata_dir, patient_id, mode='train'):
    fs = 256
    SOP = 30 * 60 * fs 
    """
    المرحلة الرابعة: فصل البيانات وتحميلها بناءً على segmentation.csv.
    mode: 'train' (تحميل الملفات المسمومة بـ 1) أو 'test' (تحميل الملفات المسمومة بـ 3).
    """
    # 1. تحديد القيمة المطلوب البحث عنها في ملف السجمينتيشن
    target_val = 1 if mode == 'train' else 3
    
    # 2. تحميل ملف segmentation.csv وملف ملخص النوبات
    seg_df = pd.read_csv(os.path.join(metadata_dir, 'segmentation.csv'), header=None, names=['filename', 'label'])
    summary_df = pd.read_csv(os.path.join(metadata_dir, 'seizure_summary.csv'))
    
    final_X = []
    final_y = []
    
    # 3. الفلترة حسب المريض والنمط (تدريب أو اختبار)
    # ملاحظة: ملف segmentation يحتوي على أسماء ملفات مثل chb01_01.edf
    patient_prefix = f"chb{int(patient_id):02d}"
    patient_files = seg_df[
    (seg_df['filename'].str.startswith(patient_prefix)) &
    (seg_df['label'] == target_val)
]
    
    for _, row in patient_files.iterrows():
        fname = row['filename']
        
        # فحص هل الملف يحتوي على نوبة (Preictal) أم هو حالة سلم (Interictal)
        # البحث في ملخص النوبات عن هذا الملف
        seizure_info = summary_df[summary_df['File_name'] == fname]

        if not seizure_info.empty:
            label = 1
            
            prev_sp = -1e12  # لمنع التداخل بين النوبات
            
            for _, sz_row in seizure_info.iterrows():
                sz_start = sz_row['Seizure_start']
                sz_stop = sz_row['Seizure_stop']
                
                data = load_preictal_segment(data_dir, patient_id, fname, sz_start)
                
                if data is None:
                    continue
                
                fs = 256
                SOP = 30 * 60 * fs
                
                # تحقق من طول العينة (نفس القديم)
                if data.shape[0] != SOP:
                    continue
                
                # حساب st و sp لنفس منطق القديم
                st = int(sz_start * fs) - 5 * 60 * fs - SOP
                sp = int(sz_stop * fs)
                
                # منع التداخل
                if st > prev_sp:
                    prev_sp = sp
                    
                    scaled_data = apply_scaling(data)
                    X_windows, y_windows = create_windows(scaled_data, label)
                    
                    if len(X_windows) > 0:
                        final_X.append(X_windows)
                        final_y.append(y_windows)
                else:
                    prev_sp = sp
        else:
            # حالة Interictal: نطبق منطق الاستثناءات والـ Notch filter
            label = 0
    
            data = load_interictal_segment(data_dir, metadata_dir, patient_id, fname)
            
            if data is None or len(data) == 0:
                continue
            
            fs = 256
            min_len = 30 * fs
            if data.shape[0] < min_len:
                continue

            scaled_data = apply_scaling(data)
            X_windows, y_windows = create_windows(scaled_data, label)           
            if len(X_windows) > 0:
                final_X.append(X_windows)
                final_y.append(y_windows)
            

    # تحويل القوائم إلى مصفوفات Numpy نهائية جاهزة للنموذج
    if not final_X:
        return np.array([]), np.array([])
        
    X = np.concatenate(final_X, axis=0)
    y = np.concatenate(final_y, axis=0)

    # ✅ توحيد عدد العينات (مثل القديم)
    unique_labels = np.unique(y)

    min_count = min([np.sum(y == lbl) for lbl in unique_labels])

    X_balanced = []
    y_balanced = []

    for lbl in unique_labels:
        idx = np.where(y == lbl)[0][:min_count]
        X_balanced.append(X[idx])
        y_balanced.append(y[idx])

    X = np.concatenate(X_balanced, axis=0)
    y = np.concatenate(y_balanced, axis=0)

    print("Balanced dataset:", X.shape)
    print("Label distribution:", np.unique(y, return_counts=True))

    return X, y