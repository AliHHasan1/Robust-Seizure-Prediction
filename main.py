import os
import numpy as np
import tensorflow as tf
import argparse
import warnings
import sys
import gc

# استيراد الأجزاء المحدثة من المجلدات التنظيمية
from utils.load_signals import prepare_dataset_by_mode 
from models.helping_functions import calc_metrics, collect_results
from utils.load_results import summary_results, load_results
from utils.save_load import savefile, load_hickle_file
from models.model import Robust_CNN_GRU 

if not sys.warnoptions:
    warnings.simplefilter("ignore")

def get_args():
    parser = argparse.ArgumentParser(description='Robust Seizure Prediction with TF2')
    parser.add_argument('-d', '--dataset', type=str, default='CHBMIT', help='Dataset name')
    parser.add_argument('-p_id', '--patient_id', type=str, default='1', help='Patient ID (e.g., 1, 2, 3...)')
    parser.add_argument('-m', '--mode', type=str, default='AE', choices=['AE', 'without_AE'], help='Training mode')
    parser.add_argument('-b', '--batch_size', type=int, default=24, help='Effective batch size')
    parser.add_argument('--micro_batch_size', type=int, default=6, help='Micro-batch size per GPU step')
    parser.add_argument('--adv_batch_size', type=int, default=2, help='Batch size for adversarial sample generation')
    parser.add_argument('--adv_steps', type=int, default=10, help='Iterations for adversarial sample generation')
    parser.add_argument('-e_init', '--epochs_initial', type=int, default=50, help='Initial training epochs')
    parser.add_argument('-e_adv', '--epochs_adversarial', type=int, default=15, help='Adversarial training epochs')
    parser.add_argument('-perc', '--percentage', type=float, default=0.4, help='Percentage of AE data')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--rebuild_cache', action='store_true', help='Force rebuilding dataset cache')
    return parser.parse_args()

def configure_gpu():
    """تقليل احتمال OOM عبر تفعيل memory growth على كل GPU متاح."""
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        return
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"Could not enable memory growth: {e}")

def run_training(args):
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # 1. إعداد المسارات (يجب تعديلها لتناسب جهازك E:\...)
    DATA_DIR = r"E:\Graduation project\DATA\chb-mit-scalp-eeg-database-1.0.0\chb-mit-scalp-eeg-database-1.0.0"
    METADATA_DIR = "E:/Graduation project/Robust-Seizure-Prediction/data_configs"
    RESULTS_DIR = f"results/results_{args.dataset}_{args.mode}/"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"\n--- Starting Training for Patient {args.patient_id} ---")
    
    # 2. تحميل البيانات (Preictal & Interictal)
    # ملاحظة: في المنهجية العلمية، نحتاج لفصل النوبات لعمل Cross-Validation
    # سنقوم هنا بتحميل البيانات كاملة ثم تقسيمها داخل حلقة الـ CV
    X, y, groups = prepare_dataset_by_mode(
        DATA_DIR,
        METADATA_DIR,
        args.patient_id,
        mode='train',
        return_groups=True,
        rebuild_cache=args.rebuild_cache,
        seed=args.seed
    )
    
    if X.size == 0:
        print(f"No data found for patient {args.patient_id}. Skipping...")
        return

    # 3. إعداد هيكل النتائج (History)
    history = {
        'test_sensitivity': [],
        'test_fpr': [],
        'test_accuracy': [],
        'AUC_AVG': [],
        'y_pred': [],
        'y_test': []
    }

    # 4. تطبيق Cross-Validation بنفس منطق التقسيم الحالي (3 folds)
    
    y_int = np.asarray(y, dtype=np.int32)
    ictal_indices = np.where(y_int == 1)[0]
    interictal_indices = np.where(y_int == 0)[0]

    ictal_group_ids = np.unique(groups[ictal_indices])
    ictal_group_ids = ictal_group_ids[ictal_group_ids >= 0]
    n_folds = len(ictal_group_ids)
    if n_folds < 2:
        raise ValueError("Not enough seizures to perform LOSO cross-validation.")

    rng = np.random.default_rng(args.seed)
    inter_perm = interictal_indices.copy()
    rng.shuffle(inter_perm)
    interictal_folds = np.array_split(inter_perm, n_folds)

    fold = 1
    for i, test_group in enumerate(ictal_group_ids):
        test_ictal_idx = np.where((y_int == 1) & (groups == test_group))[0]
        test_interictal_pool = interictal_folds[i]
        test_interictal_idx = test_interictal_pool[:len(test_ictal_idx)]

        train_ictal_idx = np.where((y_int == 1) & (groups != test_group))[0]
        train_interictal_idx = np.concatenate([interictal_folds[j] for j in range(n_folds) if j != i], axis=0)

        # موازنة interictal في التدريب مثل المنطق الأصلي
        train_interictal_idx = train_interictal_idx[:len(train_ictal_idx)]

        split_idx = int(len(train_ictal_idx) * 0.9)
        train_idx = np.concatenate([train_ictal_idx[:split_idx], train_interictal_idx[:split_idx]], axis=0)
        val_idx = np.concatenate([train_ictal_idx[split_idx:], train_interictal_idx[split_idx:]], axis=0)
        test_idx = np.concatenate([test_ictal_idx, test_interictal_idx], axis=0)

        rng.shuffle(train_idx)
        rng.shuffle(val_idx)
        
        print(f"\n--- Fold {fold} ---")

        X_train = np.asarray(X[train_idx], dtype=np.float32)
        y_train = tf.keras.utils.to_categorical(y_int[train_idx], num_classes=2)
        X_val = np.asarray(X[val_idx], dtype=np.float32)
        y_val = tf.keras.utils.to_categorical(y_int[val_idx], num_classes=2)
        X_test = np.asarray(X[test_idx], dtype=np.float32)
        y_test = tf.keras.utils.to_categorical(y_int[test_idx], num_classes=2)

        input_dim = (X_train.shape[1], X_train.shape[2]) # (7168, 18)
        current_batch = args.batch_size

        while True:
            print(f"Training with batch_size={current_batch}")
            model_wrapper = Robust_CNN_GRU(dim=input_dim, noise_limit=0.3, l2_reg=0.001)

            try:
                # التدريب (مع أو بدون AE)
                if args.mode == 'AE':
                    model_wrapper.train_with_adversarial(
                        X_train, y_train,
                        epochs_initial=args.epochs_initial,
                        epochs_adversarial=args.epochs_adversarial,
                        batch_size=current_batch,
                        percentage=args.percentage,
                        micro_batch_size=args.micro_batch_size,
                        adv_batch_size=args.adv_batch_size,
                        adv_steps=args.adv_steps
                    )
                else:
                    # تدريب عادي فقط
                    model_wrapper.model.compile(
                        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipvalue=1.0),
                        loss='categorical_crossentropy',
                        metrics=['accuracy']
                    )
                    model_wrapper.model.fit(
                        X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=args.epochs_initial,
                        batch_size=current_batch
                    )
                break
            except tf.errors.ResourceExhaustedError:
                if current_batch <= 8:
                    raise
                next_batch = max(8, current_batch // 2)
                print(f"OOM detected. Reducing batch size from {current_batch} to {next_batch} and retrying...")
                current_batch = next_batch
                tf.keras.backend.clear_session()
                gc.collect()

        # التقييم على بيانات الاختبار (المستبعدة في هذا الـ Fold)
        y_pred_probs = model_wrapper.model.predict(X_test)
        metrics = calc_metrics(y_test, y_pred_probs)
        
        # تجميع النتائج
        history = collect_results(history, metrics, y_pred_probs, y_test)
        
        print(f"Fold {fold} Results - Sensitivity: {metrics[2]*100:.2f}%, FPR: {metrics[3]:.4f}, AUC: {metrics[0]:.4f}")
        del model_wrapper
        tf.keras.backend.clear_session()
        gc.collect()
        fold += 1

    # 5. حفظ النتائج النهائية للمريض
    save_path = os.path.join(RESULTS_DIR, f"history_{args.patient_id}")
    savefile(history, save_path)
    print(f"\nResults for Patient {args.patient_id} saved to {save_path}.pkl")

def main():
    configure_gpu()
    args = get_args()
    run_training(args)
    
    # عرض ملخص النتائج (إذا كان هناك نتائج سابقة)
    # summary_results([args.patient_id], load_results(f"results/results_{args.dataset}_{args.mode}/", args.dataset)[0])

if __name__ == "__main__":
    main()
