import os
import numpy as np
import tensorflow as tf
import argparse
import warnings
import sys
import gc

# استيراد الأجزاء المحدثة من المجلدات التنظيمية
from utils.load_signals import PrepData
from models.helping_functions import train_val_cv_split, calc_metrics, collect_results
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
    parser.add_argument('--adv_steps', type=int, default=100, help='Iterations for adversarial sample generation')
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


def data_loading(target, settings):
    """
    واجهة متوافقة مع المستودع القديم: تعيد ictal/interictal folds للمريض.
    """
    print('Data Loading...................................')
    ictal_X, ictal_y = PrepData(target, type='ictal', settings=settings).apply()
    interictal_X, interictal_y = PrepData(target, type='interictal', settings=settings).apply()
    return ictal_X, ictal_y, interictal_X, interictal_y

def train(args):
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    target = args.patient_id

    # 1. إعداد المسارات (يجب تعديلها لتناسب جهازك E:\...)
    DATA_DIR = r"E:\Graduation project\DATA\chb-mit-scalp-eeg-database-1.0.0\chb-mit-scalp-eeg-database-1.0.0"
    METADATA_DIR = "E:/Graduation project/Robust-Seizure-Prediction/data_configs"
    RESULTS_DIR = f"results/results_{args.dataset}_{args.mode}/"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    settings = {
        "dataset": args.dataset,
        "datadir": DATA_DIR,
        "metadata_dir": METADATA_DIR,
        "rebuild_cache": args.rebuild_cache
    }

    print(f"\n--- Starting Training for Patient {target} ---")
    
    # 2. تحميل البيانات بنمط TF1 (folds مستقلة للنوبات)
    ictal_X, ictal_y, interictal_X, interictal_y = data_loading(target, settings)

    if len(ictal_X) < 2 or len(interictal_X) < 2:
        print(f"Insufficient TF1-style folds for patient {target}. Skipping...")
        return settings

    # 3. إعداد هيكل النتائج (History)
    history = {
        'test_sensitivity': [],
        'test_fpr': [],
        'test_accuracy': [],
        'AUC_AVG': [],
        'y_pred': [],
        'y_test': []
    }

    # 4. LOSO CV بنفس helper الخاص بالنسخة الأصلية
    fold = 1
    for X_train, y_train_raw, X_val, y_val_raw, X_test, y_test_raw in train_val_cv_split(
        ictal_X, ictal_y, interictal_X, interictal_y, val_ratio=0.1, is_shuffling=True
    ):
        print(f"\n--- Fold {fold} ---")

        X_train = np.asarray(X_train, dtype=np.float32)
        y_train = tf.keras.utils.to_categorical(np.asarray(y_train_raw, dtype=np.int32), num_classes=2)
        X_val = np.asarray(X_val, dtype=np.float32)
        y_val = tf.keras.utils.to_categorical(np.asarray(y_val_raw, dtype=np.int32), num_classes=2)
        X_test = np.asarray(X_test, dtype=np.float32)
        y_test = tf.keras.utils.to_categorical(np.asarray(y_test_raw, dtype=np.int32), num_classes=2)

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
                        x_val=X_val, y_val=y_val,
                        epochs_initial=args.epochs_initial,
                        epochs_adversarial=args.epochs_adversarial,
                        batch_size=current_batch,
                        percentage=args.percentage,
                        micro_batch_size=args.micro_batch_size,
                        adv_batch_size=args.adv_batch_size,
                        adv_steps=args.adv_steps
                    )
                else:
                    # تدريب عادي بنمط أقرب لـ TF1
                    model_wrapper.train(
                        X_train, y_train,
                        X_val, y_val,
                        epochs=args.epochs_initial,
                        batch_size=current_batch,
                        verbose=True
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
        y_pred_probs = model_wrapper.predict(X_test)
        metrics = calc_metrics(y_test, y_pred_probs)
        
        # تجميع النتائج
        history = collect_results(history, metrics, y_pred_probs, y_test)
        
        print(f"Fold {fold} Results - Sensitivity: {metrics[2]*100:.2f}%, FPR: {metrics[3]:.4f}, AUC: {metrics[0]:.4f}")
        del model_wrapper
        tf.keras.backend.clear_session()
        gc.collect()
        fold += 1

    # 5. حفظ النتائج النهائية للمريض
    save_path = os.path.join(RESULTS_DIR, f"history_{target}")
    savefile(history, save_path)
    print(f"\nResults for Patient {target} saved to {save_path}.pkl")
    return settings

def main():
    configure_gpu()
    args = get_args()
    settings = train(args)
    
    # عرض ملخص النتائج (إذا كان هناك نتائج سابقة)
    # summary_results([args.patient_id], load_results(f"results/results_{args.dataset}_{args.mode}/", args.dataset)[0])

if __name__ == "__main__":
    main()
