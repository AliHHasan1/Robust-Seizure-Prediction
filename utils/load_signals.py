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
    
    # 1. القنوات الأساسية (Base Channels) - 18 قناة
    base_chs = [
        'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 
        'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2', 
        'FZ-CZ', 'CZ-PZ'
    ]
    # 2. القنوات الإضافية (Extra Channels) - 4 قنوات
    extra_chs = ['P7-T7', 'T7-FT9', 'FT9-FT10', 'FT10-T8']
    
    # 3. قراءة عدد القنوات المطلوب لهذا المريض من ملف الإعدادات
    sampling_path = os.path.join(metadata_dir, 'sampling_CHBMIT.csv')
    if os.path.exists(sampling_path):
        sampling_df = pd.read_csv(sampling_path)
        # البحث عن المريض في العمود Subject
        subject_info = sampling_df[sampling_df['Subject'].astype(str) == subject_id]
        if not subject_info.empty:
            num_electrodes = int(subject_info.iloc[0]['Electrode'])
            
            # إذا كان المطلوب 18 قناة، نأخذ القنوات الأساسية فقط
            if num_electrodes == 18:
                return base_chs
            # إذا كان المطلوب 22 قناة، نأخذ الأساسية + الإضافية
            elif num_electrodes == 22:
                return base_chs + extra_chs
            # حالات خاصة للمرضى 13 و 16 (17 قناة كما في المستودع الأصلي)
            elif num_electrodes == 17:
                return base_chs[:-1]
    
    # Fallback في حال عدم وجود الملف أو المريض
    if subject_id in ['13', '16']:
        return base_chs[:-1]
    return base_chs + extra_chs

def get_previous_file_name(current_file):
    """استنتاج اسم الملف السابق لربط الإشارات متصلة زمنياً"""
    try:
        # التعامل مع حالات مثل chb01_01.edf أو chb01_01+.edf
        base_name = os.path.splitext(current_file)[0]
        if '+' in base_name:
            base_name = base_name.replace('+', '')
            
        parts = base_name.split('_')
        prefix = parts[0]
        seq = int(parts[1])
        
        if seq <= 1:
            return None
        return f"{prefix}_{seq-1:02d}.edf"
    except:
        return None

def load_raw_with_fallback(file_path, channels):
    """تحميل ملف EDF مع معالجة اختلاف أسماء القنوات (Case Sensitivity)"""
    try:
        raw = read_raw_edf(file_path, preload=True, verbose=False)
        # توحيد أسماء القنوات لتجنب مشاكل الأحرف الكبيرة والصغيرة
        available_channels = raw.ch_names
        target_channels = []
        for ch in channels:
            # البحث عن القناة بغض النظر عن حالة الأحرف
            match = [a for a in available_channels if a.upper() == ch.upper()]
            if match:
                target_channels.append(match[0])
        
        if len(target_channels) > 0:
            raw.pick_channels(target_channels)
            return raw
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def load_preictal_segment(data_dir, metadata_dir, patient_id, edf_file, sz_start_sec):
    """
    اقتطاع 30 دقيقة (SOP) قبل النوبة بـ 5 دقائق (SPH).
    """
    fs = 256 
    SOP = 30 * 60 * fs 
    SPH = 5 * 60 * fs  
    
    st = int(sz_start_sec * fs) - SPH - SOP
    sp = int(sz_start_sec * fs) - SPH
    
    channels = get_channels_by_subject(metadata_dir, patient_id)
    patient_folder = f"chb{int(patient_id):02d}"
    current_file_path = os.path.join(data_dir, patient_folder, edf_file)
    
    raw = load_raw_with_fallback(current_file_path, channels)
    if raw is None: return None

    raw.notch_filter(np.arange(60, 121, 60), fir_design='firwin', verbose=False)
    current_data = raw.get_data().T 
    
    if st < 0:
        prev_file = get_previous_file_name(edf_file)
        if prev_file:
            prev_path = os.path.join(data_dir, patient_folder, prev_file)
            if os.path.exists(prev_path):
                prev_raw = load_raw_with_fallback(prev_path, channels)
                if prev_raw:
                    prev_raw.notch_filter(np.arange(60, 121, 60), fir_design='firwin', verbose=False)
                    prev_data = prev_raw.get_data().T
                    
                    # دمج البيانات من الملف السابق والحالي
                    try:
                        data = np.concatenate((prev_data[st:], current_data[:sp]), axis=0)
                        if data.shape[0] != SOP: return None
                        return data
                    except:
                        return None
        
        # fallback إذا لم يوجد ملف سابق أو فشل التحميل
        if sp > 0:
            data = current_data[0:sp]
            # إذا كانت البيانات أقل من SOP، نقوم بالحشو بالأصفار أو تجاهلها (هنا نتجاهلها للحفاظ على الدقة)
            if data.shape[0] != SOP: return None
            return data
        return None

    else:
        data = current_data[st:sp]
        if data.shape[0] != SOP: return None
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
    channels = get_channels_by_subject(metadata_dir,patient_id)
    patient_folder = f"chb{int(patient_id):02d}"
    file_path = os.path.join(data_dir, patient_folder, edf_file)
    
    # 1. تحميل الإشارة الخام
    raw = load_raw_with_fallback(file_path, channels)
    if raw is None: return None
    
    # 2. فحص وجود استثناءات زمنية لهذا الملف
    special_df = load_special_interictal_metadata(metadata_dir)
    file_constraints = special_df[special_df['File_name'] == edf_file]
    
    if not file_constraints.empty:
        # إذا وجد قيد زمني، نقوم بقص الإشارة (مثلاً من الثانية X إلى الثانية Y)
        t_start = float(file_constraints.iloc[0]['Start'])
        t_end = float(file_constraints.iloc[0]['End'])
        
        # إذا كانت القيمة -1 تعني حتى نهاية الملف
        t_end = None if t_end == -1 else t_end
        try:
            raw.crop(tmin=t_start, tmax=t_end)
        except:
            pass
        
    raw.notch_filter(np.arange(60, 121, 60), fir_design='firwin', verbose=False)
    data = raw.get_data().T
    return data if data.shape[0] > 0 else None

  

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
    if not windows_X: return np.array([]), np.array([])
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
    seg_path = os.path.join(metadata_dir, 'segmentation.csv')
    sum_path = os.path.join(metadata_dir, 'seizure_summary.csv')
    
    if not os.path.exists(seg_path) or not os.path.exists(sum_path):
        print(f"Error: Metadata files not found in {metadata_dir}")
        return np.array([]), np.array([])

    seg_df = pd.read_csv(seg_path, header=None, names=['filename', 'label'])
    summary_df = pd.read_csv(sum_path)
    
    final_X = []
    final_y = []
    
    patient_prefix = f"chb{int(patient_id):02d}"
    patient_files = seg_df[(seg_df['filename'].str.startswith(patient_prefix)) & (seg_df['label'] == target_val)]
    
    print(f"Processing {len(patient_files)} files for patient {patient_id} ({mode} mode)...")

    for _, row in patient_files.iterrows():
        fname = row['filename']
        
        # فحص هل الملف يحتوي على نوبة (Preictal) أم هو حالة سلم (Interictal)
        # البحث في ملخص النوبات عن هذا الملف
        seizure_info = summary_df[summary_df['File_name'] == fname]

        if not seizure_info.empty:
            # حالة Preictal
            prev_sp = -1e12
            for _, sz_row in seizure_info.iterrows():
                sz_start = sz_row['Seizure_start']
                data = load_preictal_segment(data_dir, patient_id, fname, sz_start)
                
                if data is not None:
                    scaled_data = apply_scaling(data)
                    X_w, y_w = create_windows(scaled_data, 1)
                    if X_w.size > 0:
                        final_X.append(X_w)
                        final_y.append(y_w)
        else:
            # حالة Interictal
            data = load_interictal_segment(data_dir, metadata_dir, patient_id, fname)
            if data is not None and len(data) >= (30 * 256):
                scaled_data = apply_scaling(data)
                X_w, y_w = create_windows(scaled_data, 0)
                if X_w.size > 0:
                    final_X.append(X_w)
                    final_y.append(y_w)

    if not final_X:
        return np.array([]), np.array([])
        
    X = np.concatenate(final_X, axis=0)
    y = np.concatenate(final_y, axis=0)

    # موازنة الفئات (Class Balancing)
    unique_labels, counts = np.unique(y, return_counts=True)
    if len(unique_labels) > 1:
        min_count = min(counts)
        X_balanced, y_balanced = [], []
        for lbl in unique_labels:
            idx = np.where(y == lbl)[0]
            np.random.shuffle(idx) # خلط العينات قبل الاختيار
            idx = idx[:min_count]
            X_balanced.append(X[idx])
            y_balanced.append(y[idx])
        X = np.concatenate(X_balanced, axis=0)
        y = np.concatenate(y_balanced, axis=0)

    print(f"Final Balanced Dataset Shape: {X.shape}")
    return X, y