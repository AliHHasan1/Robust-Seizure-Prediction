import os
import numpy as np
import tensorflow as tf
import pickle

# استيراد الأجزاء المحدثة من المجلدات التنظيمية
from utils.load_signals import prepare_dataset_by_mode 
from utils.load_results import summary_results, load_results
from models.model import Robust_CNN_GRU 

def run_training_pipeline(patient_id):
    # 1. إعداد المسارات (تأكد أن المجلدات موجودة)
    # مسار البيانات الضخمة (ملفات EDF)
    DATA_DIR = r'E:\Graduation project\DATA\chb-mit-scalp-eeg-database-1.0.0\chb-mit-scalp-eeg-database-1.0.0' 
    
    # مسار مجلد الإعدادات الجديد (ملفات CSV) الذي أنشأته داخل المشروع
    METADATA_DIR = os.path.join(os.getcwd(), 'data_configs')
    
    # مسار حفظ النتائج
    RESULT_DIR = r'E:\Graduation project\Robust-Seizure-Prediction\results'
    os.makedirs(RESULT_DIR, exist_ok=True)
    
    print(f"\n--- بدء العمل على المريض رقم {patient_id} ---")

    # 2. تحميل ومعالجة البيانات
    # استخدام دالة prepare_dataset_by_mode التي تم تحليلها وتصحيحها
    print("جاري تحضير بيانات التدريب...")
    X_train, y_train = prepare_dataset_by_mode(DATA_DIR, METADATA_DIR, patient_id, mode='train')
    
    if X_train.size == 0:
        print(f"لا توجد بيانات تدريب للمريض {patient_id}. يرجى التحقق من المسارات والملفات.")
        return

    print(f"أبعاد بيانات التدريب (X_train): {X_train.shape}")
    print(f"أبعاد تسميات التدريب (y_train): {y_train.shape}")

    # تحويل التسميات إلى One-Hot Encoding
    y_train_categorical = tf.keras.utils.to_categorical(y_train, num_classes=2)
    print(f"أبعاد تسميات التدريب بعد One-Hot Encoding (y_train_categorical): {y_train_categorical.shape}")

    # 3. بناء النموذج الحديث
    input_shape = (X_train.shape[1], X_train.shape[2]) # (الزمن، القنوات)
    # تأكد من تمرير المعاملات الصحيحة للـ Robust_CNN_GRU (خاصة l2_reg و lamda)
    model_wrapper = Robust_CNN_GRU(dim=input_shape, noise_limit=0.02, l2_reg=0.001, lamda=1.0) 
    model_wrapper.model.summary()

    # 4. حلقة التدريب العدائي
    # استخدام دالة train_with_adversarial المدمجة في النموذج
    print("بدء التدريب العدائي (Adversarial Training) باستخدام الدالة المخصصة...")
    batch_size = 32
    epochs_initial = 50 # عدد الـ epochs للتدريب الأولي (كما في الكود القديم)
    epochs_adversarial = 15 # عدد الـ epochs للتدريب مع الأمثلة العدائية (كما في الكود القديم)
    adversarial_percentage = 0.3 # نسبة الأمثلة العدائية (كما في الكود القديم)
    
    # دالة train_with_adversarial في النموذج المصحح تقوم بكل الخطوات داخلياً
    # بما في ذلك التدريب الأولي، توليد الأمثلة العدائية، والتدريب عليها
    model_wrapper.train_with_adversarial(
        x_train=X_train,
        y_train=y_train_categorical,
        epochs_initial=epochs_initial, # تم إضافة هذا المعامل في النسخة المصححة من model.py
        epochs_adversarial=epochs_adversarial, # تم إضافة هذا المعامل في النسخة المصححة من model.py
        batch_size=batch_size,
        percentage=adversarial_percentage
    )

    # 5. حفظ النتائج والنموذج
    model_path = os.path.join(RESULT_DIR, f'model_patient_{patient_id}.h5')
    model_wrapper.model.save(model_path)
    print(f"تم حفظ النموذج في: {model_path}")
    
    # يمكنك إضافة خطوات التقييم هنا بعد انتهاء التدريب
    # على سبيل المثال، تقييم النموذج على بيانات الاختبار (X_test, y_test)
    # y_pred = model_wrapper.model.predict(X_test)
    # ... حساب المقاييس وحفظها ...

    print(f"اكتملت عملية التدريب والحفظ للمريض رقم {patient_id} بنجاح!")

if __name__ == "__main__":
    # تشغيل التدريب للمريض رقم 1
    run_training_pipeline(patient_id=1)