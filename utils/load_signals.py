import os
import numpy as np
import pandas as pd
import mne
import shutil
import tempfile
import pickle
from mne.io import read_raw_edf
from sklearn.preprocessing import StandardScaler

def get_channels_by_subject(metadata_dir, subject_id):
    """
    Channel map aligned with the original TF1 repository.
    """
    subject_id = str(subject_id)
    base_18 = [
        'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
        'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
        'FZ-CZ', 'CZ-PZ'
    ]
    extra_4 = ['P7-T7', 'T7-FT9', 'FT9-FT10', 'FT10-T8']

    # Original special handling: patients 13/16 use 17 channels.
    if subject_id in ['13', '16']:
        return [
            'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
            'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8', 'FZ-CZ',
            'CZ-PZ'
        ]

    # Original special handling: patient 4 uses a custom 20-channel set.
    if subject_id == '4':
        return [
            'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
            'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'P8-O2', 'FZ-CZ',
            'CZ-PZ', 'P7-T7', 'T7-FT9', 'FT10-T8'
        ]

    return base_18 + extra_4

def get_previous_file_name(current_file):
    """
    Previous EDF naming aligned with original TF1 logic, including chb02_16+.edf.
    """
    try:
        if current_file == 'chb02_16+.edf':
            return 'chb02_16.edf'
        if current_file[6] == '_':
            seq = int(current_file[7:9])
            return f"{current_file[:6]}_{seq-1:02d}.edf"
        seq = int(current_file[6:8])
        return f"{current_file[:5]}_{seq-1:02d}.edf"
    except Exception:
        return None

def load_raw_with_fallback(file_path, channels):
    """Load EDF and pick channels with TF1-like strict channel names."""
    try:
        raw = read_raw_edf(file_path, preload=True, verbose=False)
        raw.pick(channels)
        return raw
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def load_preictal_segment(data_dir, metadata_dir, patient_id, edf_file, sz_start, sz_stop=None, prev_sp=-1e12):
    """
    اقتطاع 30 دقيقة (SOP) قبل النوبة بـ 5 دقائق (SPH).
    """
    fs = 256 
    SOP = 30 * 60 * fs 
    SPH = 5 * 60 * fs  
    
    st = int(sz_start * fs) - SPH
    sp = int(sz_stop * fs) if sz_stop is not None else st
    if (st - SOP) <= prev_sp:
        return None
    
    channels = get_channels_by_subject(metadata_dir, patient_id)
    patient_folder = f"chb{int(patient_id):02d}"
    current_file_path = os.path.join(data_dir, patient_folder, edf_file)
    
    raw = load_raw_with_fallback(current_file_path, channels)
    if raw is None: return None

    raw.notch_filter(np.arange(60, 121, 60), fir_design='firwin', verbose=False)
    current_data = raw.get_data().T 
    
    if st - SOP < 0:
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
                        data = np.concatenate((prev_data[st - SOP:], current_data[:st]), axis=0)
                        if data.shape[0] != SOP: return None
                        return data
                    except:
                        return None
        
        # fallback إذا لم يوجد ملف سابق أو فشل التحميل
        if st > 0:
            data = current_data[:st]
            # إذا كانت البيانات أقل من SOP، نقوم بالحشو بالأصفار أو تجاهلها (هنا نتجاهلها للحفاظ على الدقة)
            if data.shape[0] != SOP: return None
            return data
        return None

    else:
        data = current_data[st - SOP:st]
        if data.shape[0] != SOP: return None
        return data


def _is_patient_file_match(filename, patient_id):
    patient_id = str(patient_id)
    return (
        (f"chb{patient_id}_" in filename) or
        (f"chb0{patient_id}_" in filename) or
        (f"chb{patient_id}a_" in filename) or
        (f"chb{patient_id}b_" in filename) or
        (f"chb{patient_id}c_" in filename)
    )


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

def prepare_dataset_by_mode(data_dir, metadata_dir, patient_id, mode='train', return_groups=False, rebuild_cache=False, seed=42):
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
    
    cache_root = os.path.join('results', 'dataset_cache')
    patient_cache_dir = os.path.join(cache_root, f"chb{int(patient_id):02d}_{mode}")
    os.makedirs(patient_cache_dir, exist_ok=True)

    X_cache_path = os.path.join(patient_cache_dir, 'X_balanced.npy')
    y_cache_path = os.path.join(patient_cache_dir, 'y_balanced.npy')
    g_cache_path = os.path.join(patient_cache_dir, 'g_balanced.npy')

    # إعادة استخدام الكاش لتجنب إعادة المعالجة الثقيلة
    if (not rebuild_cache) and os.path.exists(X_cache_path) and os.path.exists(y_cache_path):
        X_cached = np.load(X_cache_path, mmap_mode='r')
        y_cached = np.load(y_cache_path, mmap_mode='r')
        print(f"Loaded cached dataset: {X_cached.shape} from {patient_cache_dir}")
        if return_groups:
            if os.path.exists(g_cache_path):
                g_cached = np.load(g_cache_path, mmap_mode='r')
                return X_cached, y_cached, g_cached
            # كاش قديم بدون group ids
            print("Cached dataset missing seizure groups. Rebuilding cache...")
        else:
            return X_cached, y_cached

    # تخزين النوافذ على القرص مؤقتاً بدل RAM
    build_dir = tempfile.mkdtemp(prefix='build_', dir=patient_cache_dir)
    chunk_records = {0: [], 1: []}
    chunk_counter = 0
    rng = np.random.default_rng(seed)
    seizure_group_id = 0
    
    patient_files = seg_df[
        seg_df['filename'].apply(lambda x: _is_patient_file_match(str(x), patient_id)) &
        (seg_df['label'] == target_val)
    ]
    
    print(f"Processing {len(patient_files)} files for patient {patient_id} ({mode} mode)...")

    for _, row in patient_files.iterrows():
        fname = row['filename']
        
        # فحص هل الملف يحتوي على نوبة (Preictal) أم هو حالة سلم (Interictal)
        # البحث في ملخص النوبات عن هذا الملف
        seizure_info = summary_df[summary_df['File_name'] == fname]

        if not seizure_info.empty:
            # حالة Preictal
            prev_sp = -1e6
            for _, sz_row in seizure_info.iterrows():
                sz_start = sz_row['Seizure_start']
                sz_stop = sz_row['Seizure_stop'] if 'Seizure_stop' in sz_row else None
                data = load_preictal_segment(
                    data_dir, metadata_dir, patient_id, fname, sz_start, sz_stop=sz_stop, prev_sp=prev_sp
                )
                if sz_stop is not None:
                    prev_sp = int(sz_stop * 256)
                
                if data is not None:
                    scaled_data = apply_scaling(data)
                    X_w, y_w = create_windows(scaled_data, 1)
                    if X_w.size > 0:
                        chunk_path = os.path.join(build_dir, f'chunk_{chunk_counter:06d}.npy')
                        group_path = os.path.join(build_dir, f'group_{chunk_counter:06d}.npy')
                        np.save(chunk_path, X_w)
                        np.save(group_path, np.full((len(y_w),), seizure_group_id, dtype='int32'))
                        chunk_records[1].append({'path': chunk_path, 'group_path': group_path, 'count': int(len(y_w))})
                        chunk_counter += 1
                        seizure_group_id += 1
        else:
            # حالة Interictal
            data = load_interictal_segment(data_dir, metadata_dir, patient_id, fname)
            if data is not None and len(data) >= (30 * 256):
                scaled_data = apply_scaling(data)
                X_w, y_w = create_windows(scaled_data, 0)
                if X_w.size > 0:
                    chunk_path = os.path.join(build_dir, f'chunk_{chunk_counter:06d}.npy')
                    group_path = os.path.join(build_dir, f'group_{chunk_counter:06d}.npy')
                    np.save(chunk_path, X_w)
                    np.save(group_path, np.full((len(y_w),), -1, dtype='int32'))
                    chunk_records[0].append({'path': chunk_path, 'group_path': group_path, 'count': int(len(y_w))})
                    chunk_counter += 1

    total_0 = sum(r['count'] for r in chunk_records[0])
    total_1 = sum(r['count'] for r in chunk_records[1])

    if total_0 == 0 or total_1 == 0:
        shutil.rmtree(build_dir, ignore_errors=True)
        return np.array([]), np.array([])

    min_count = min(total_0, total_1)
    print(f"Balancing classes on disk: class0={total_0}, class1={total_1}, target={min_count}")

    # استنتاج الأبعاد من أول chunk متاح
    probe_path = chunk_records[0][0]['path'] if chunk_records[0] else chunk_records[1][0]['path']
    probe = np.load(probe_path, mmap_mode='r')
    _, win_len, n_channels = probe.shape

    X_balanced = np.lib.format.open_memmap(
        X_cache_path, mode='w+', dtype='float32', shape=(min_count * 2, win_len, n_channels)
    )
    y_balanced = np.lib.format.open_memmap(
        y_cache_path, mode='w+', dtype='float32', shape=(min_count * 2,)
    )
    g_balanced = np.lib.format.open_memmap(
        g_cache_path, mode='w+', dtype='int32', shape=(min_count * 2,)
    )

    def write_class_samples(records, class_label, write_pos):
        total = sum(r['count'] for r in records)
        sampled_idx = np.arange(total)
        rng.shuffle(sampled_idx)
        sampled_idx = np.sort(sampled_idx[:min_count])

        ptr = 0
        offset = 0
        for rec in records:
            start = offset
            end = offset + rec['count']
            local_mask = (sampled_idx >= start) & (sampled_idx < end)
            local_idx = sampled_idx[local_mask] - start

            if len(local_idx) > 0:
                chunk = np.load(rec['path'], mmap_mode='r')
                groups = np.load(rec['group_path'], mmap_mode='r')
                take = chunk[local_idx]
                take_g = groups[local_idx]
                next_pos = write_pos + len(take)
                X_balanced[write_pos:next_pos] = take
                y_balanced[write_pos:next_pos] = class_label
                g_balanced[write_pos:next_pos] = take_g
                write_pos = next_pos
                ptr += len(take)

            offset = end

        return write_pos, ptr

    pos = 0
    pos, wrote_0 = write_class_samples(chunk_records[0], 0.0, pos)
    pos, wrote_1 = write_class_samples(chunk_records[1], 1.0, pos)

    if wrote_0 != min_count or wrote_1 != min_count:
        shutil.rmtree(build_dir, ignore_errors=True)
        raise RuntimeError("Disk balancing failed: unexpected sample count.")

    X_balanced.flush()
    y_balanced.flush()
    g_balanced.flush()

    shutil.rmtree(build_dir, ignore_errors=True)

    X_out = np.load(X_cache_path, mmap_mode='r')
    y_out = np.load(y_cache_path, mmap_mode='r')
    g_out = np.load(g_cache_path, mmap_mode='r')
    print(f"Final Balanced Dataset Shape: {X_out.shape} (cached at {patient_cache_dir})")
    if return_groups:
        return X_out, y_out, g_out
    return X_out, y_out


def prepare_dataset_tf1_style_cv(data_dir, metadata_dir, patient_id, rebuild_cache=False):
    """
    تجهيز بيانات المريض بنمط قريب جداً من TF1:
    - ictal folds: fold لكل نوبة (preictal segment لكل seizure event).
    - interictal folds: كل interictal windows تُجمع ثم تُقسم إلى نفس عدد ictal folds.
    - بدون global balancing مسبق.
    """
    cache_root = os.path.join('results', 'dataset_cache')
    patient_cache_dir = os.path.join(cache_root, f"chb{int(patient_id):02d}_tf1style_cv")
    os.makedirs(patient_cache_dir, exist_ok=True)
    cache_file = os.path.join(patient_cache_dir, 'folds.pkl')

    if (not rebuild_cache) and os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            cached = pickle.load(f)
        print(f"Loaded TF1-style cached folds from {cache_file}")
        return cached['ictal_X'], cached['ictal_y'], cached['interictal_X'], cached['interictal_y']

    seg_path = os.path.join(metadata_dir, 'segmentation.csv')
    sum_path = os.path.join(metadata_dir, 'seizure_summary.csv')
    if not os.path.exists(seg_path) or not os.path.exists(sum_path):
        print(f"Error: Metadata files not found in {metadata_dir}")
        return [], [], [], []

    seg_df = pd.read_csv(seg_path, header=None, names=['filename', 'label'])
    summary_df = pd.read_csv(sum_path)
    # TF1: ictal files تأتي من seizure_summary مباشرة مع نفس مطابقة الأسماء القديمة.
    patient_seizures = summary_df[summary_df['File_name'].apply(lambda x: _is_patient_file_match(str(x), patient_id))]
    ictal_X_folds = []
    ictal_y_folds = []

    # كل seizure event => fold مستقل
    for fname, seizures_in_file in patient_seizures.groupby('File_name', sort=False):
        prev_sp = -1e6
        for _, sz_row in seizures_in_file.iterrows():
            sz_start = sz_row['Seizure_start']
            sz_stop = sz_row['Seizure_stop'] if 'Seizure_stop' in sz_row else None
            data = load_preictal_segment(
                data_dir, metadata_dir, patient_id, fname, sz_start, sz_stop=sz_stop, prev_sp=prev_sp
            )
            if sz_stop is not None:
                prev_sp = int(sz_stop * 256)
            if data is None:
                continue

            scaled = apply_scaling(data)
            X_w, y_w = create_windows(scaled, 1)
            if X_w.size == 0:
                continue

            ictal_X_folds.append(X_w.astype('float32'))
            ictal_y_folds.append(y_w.astype('float32'))

    if len(ictal_X_folds) < 2:
        print(f"Insufficient ictal folds for patient {patient_id}.")
        return [], [], [], []

    n_folds = len(ictal_X_folds)

    # TF1: interictal files من segmentation label=0
    patient_inter_files = seg_df[
        seg_df['filename'].apply(lambda x: _is_patient_file_match(str(x), patient_id)) &
        (seg_df['label'] == 0)
    ]['filename'].tolist()

    inter_chunks = []
    for fname in patient_inter_files:
        data = load_interictal_segment(data_dir, metadata_dir, patient_id, fname)
        if data is None or len(data) < (30 * 256):
            continue
        scaled = apply_scaling(data)
        X_w, y_w = create_windows(scaled, 0)
        if X_w.size == 0:
            continue
        inter_chunks.append((X_w.astype('float32'), y_w.astype('float32')))

    if len(inter_chunks) == 0:
        print(f"No interictal windows found for patient {patient_id}.")
        return [], [], [], []

    interictal_X_all = np.concatenate([c[0] for c in inter_chunks], axis=0)
    interictal_y_all = np.concatenate([c[1] for c in inter_chunks], axis=0)

    # تقسيم interictal إلى نفس عدد folds كما في TF1 helpers
    inter_idx_splits = np.array_split(np.arange(len(interictal_X_all)), n_folds)
    interictal_X_folds = [interictal_X_all[idx] for idx in inter_idx_splits if len(idx) > 0]
    interictal_y_folds = [interictal_y_all[idx] for idx in inter_idx_splits if len(idx) > 0]

    # لضمان التوافق مع train_val_cv_split (TF1 يعتمد min بين العددين)
    n_final = min(len(ictal_X_folds), len(interictal_X_folds))
    ictal_X_folds = ictal_X_folds[:n_final]
    ictal_y_folds = ictal_y_folds[:n_final]
    interictal_X_folds = interictal_X_folds[:n_final]
    interictal_y_folds = interictal_y_folds[:n_final]

    with open(cache_file, 'wb') as f:
        pickle.dump(
            {
                'ictal_X': ictal_X_folds,
                'ictal_y': ictal_y_folds,
                'interictal_X': interictal_X_folds,
                'interictal_y': interictal_y_folds,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL
        )
    print(f"Saved TF1-style folds cache to {cache_file}")

    return ictal_X_folds, ictal_y_folds, interictal_X_folds, interictal_y_folds


class PrepData:
    """
    واجهة متوافقة اسمياً مع المستودع القديم:
    - type='ictal'  => يعيد ictal folds
    - type='interictal' => يعيد interictal folds
    """
    def __init__(self, target, type, settings):
        self.target = str(target)
        self.type = type
        self.settings = settings

    def apply(self):
        dataset = self.settings.get('dataset', 'CHBMIT')
        if dataset != 'CHBMIT':
            raise NotImplementedError("PrepData compatibility wrapper currently supports CHBMIT only.")

        data_dir = self.settings['datadir']
        metadata_dir = self.settings['metadata_dir']
        rebuild_cache = bool(self.settings.get('rebuild_cache', False))

        ictal_X, ictal_y, interictal_X, interictal_y = prepare_dataset_tf1_style_cv(
            data_dir=data_dir,
            metadata_dir=metadata_dir,
            patient_id=self.target,
            rebuild_cache=rebuild_cache
        )

        if self.type == 'ictal':
            return ictal_X, ictal_y
        if self.type == 'interictal':
            return interictal_X, interictal_y
        raise ValueError(f"Unsupported PrepData type: {self.type}")
