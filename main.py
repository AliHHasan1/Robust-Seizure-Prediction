import os
import numpy as np
import tensorflow as tf
import argparse
import warnings
import sys

# استيراد الأجزاء المحدثة من المجلدات التنظيمية
from utils.load_signals import prepare_dataset_by_mode 
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
    parser.add_argument('-b', '--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('-e_init', '--epochs_initial', type=int, default=50, help='Initial training epochs')
    parser.add_argument('-e_adv', '--epochs_adversarial', type=int, default=15, help='Adversarial training epochs')
    parser.add_argument('-perc', '--percentage', type=float, default=0.4, help='Percentage of AE data')
    return parser.parse_args()

def run_training(args):
    # 1. إعداد المسارات (يجب تعديلها لتناسب جهازك E:\...)
    DATA_DIR = "E:/Graduation project/Robust-Seizure-Prediction/CHBMIT"
    METADATA_DIR = "E:/Graduation project/Robust-Seizure-Prediction/data_configs"
    RESULTS_DIR = f"results/results_{args.dataset}_{args.mode}/"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"\n--- Starting Training for Patient {args.patient_id} ---")
    
    # 2. تحميل البيانات (Preictal & Interictal)
    # ملاحظة: في المنهجية العلمية، نحتاج لفصل النوبات لعمل Cross-Validation
    # سنقوم هنا بتحميل البيانات كاملة ثم تقسيمها داخل حلقة الـ CV
    X, y = prepare_dataset_by_mode(DATA_DIR, METADATA_DIR, args.patient_id, mode='train')
    
    if X.size == 0:
        print(f"No data found for patient {args.patient_id}. Skipping...")
        return

    # تحويل y إلى One-hot encoding (ضروري لـ CategoricalCrossentropy)
    y_onehot = tf.keras.utils.to_categorical(y, num_classes=2)

    # 3. إعداد هيكل النتائج (History)
    history = {
        'test_sensitivity': [],
        'test_fpr': [],
        'test_accuracy': [],
        'AUC_AVG': [],
        'y_pred': [],
        'y_test': []
    }

    # 4. تطبيق الـ Cross-Validation (Leave-One-Seizure-Out)
    # ملاحظة: دالة train_val_cv_split تتوقع بيانات مقسمة حسب النوبات
    # للتبسيط هنا، سنقوم بعمل تقسيم عشوائي يحاكي الـ CV إذا لم تكن البيانات مقسمة مسبقاً
    # ولكن المنهجية الأفضل هي تمرير قائمة من النوبات (List of arrays)
    
    # تحويل البيانات إلى قائمة (List) لمحاكاة النوبات (Seizures)
    # في مشروعك الحقيقي، يفضل أن تعيد دالة prepare_dataset قائمة بالمصفوفات
    ictal_indices = np.where(y == 1)[0]
    interictal_indices = np.where(y == 0)[0]
    
    # تقسيم افتراضي لـ 3 Folds (كمثال)
    ictal_X_list = np.array_split(X[ictal_indices], 3)
    ictal_y_list = np.array_split(y_onehot[ictal_indices], 3)
    interictal_X_list = np.array_split(X[interictal_indices], 3)
    interictal_y_list = np.array_split(y_onehot[interictal_indices], 3)

    fold = 1
    for X_train, y_train, X_val, y_val, X_test, y_test in train_val_cv_split(
        ictal_X_list, ictal_y_list, interictal_X_list, interictal_y_list, val_ratio=0.1):
        
        print(f"\n--- Fold {fold} ---")
        
        # إنشاء النموذج المحدث (TF2)
        input_dim = (X_train.shape[1], X_train.shape[2]) # (7168, 18)
        model_wrapper = Robust_CNN_GRU(dim=input_dim, noise_limit=0.3, l2_reg=0.001)
        
        # التدريب (مع أو بدون AE)
        if args.mode == 'AE':
            model_wrapper.train_with_adversarial(
                X_train, y_train, 
                epochs_initial=args.epochs_initial, 
                epochs_adversarial=args.epochs_adversarial, 
                batch_size=args.batch_size, 
                percentage=args.percentage
            )
        else:
            # تدريب عادي فقط
            model_wrapper.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipvalue=1.0),
                loss='categorical_cross_entropy',
                metrics=['accuracy']
            )
            model_wrapper.model.fit(
                X_train, y_train, 
                validation_data=(X_val, y_val), 
                epochs=args.epochs_initial, 
                batch_size=args.batch_size
            )

        # التقييم على بيانات الاختبار (المستبعدة في هذا الـ Fold)
        y_pred_probs = model_wrapper.model.predict(X_test)
        metrics = calc_metrics(y_test, y_pred_probs)
        
        # تجميع النتائج
        history = collect_results(history, metrics, y_pred_probs, y_test)
        
        print(f"Fold {fold} Results - Sensitivity: {metrics[2]*100:.2f}%, FPR: {metrics[3]:.4f}, AUC: {metrics[0]:.4f}")
        fold += 1

    # 5. حفظ النتائج النهائية للمريض
    save_path = os.path.join(RESULTS_DIR, f"history_{args.patient_id}")
    savefile(history, save_path)
    print(f"\nResults for Patient {args.patient_id} saved to {save_path}.pkl")

def main():
    args = get_args()
    run_training(args)
    
    # عرض ملخص النتائج (إذا كان هناك نتائج سابقة)
    # summary_results([args.patient_id], load_results(f"results/results_{args.dataset}_{args.mode}/", args.dataset)[0])

if __name__ == "__main__":
    main()