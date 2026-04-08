import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers
import numpy as np

class CNN_GRU_Modern:
    def __init__(self, dim, noise_limit=0.02, l2_reg=0.001):
        self.win_length = dim[0]
        self.channels = dim[1]
        self.noise_limit = noise_limit
        self.l2_reg = l2_reg
        self.model = self.build_model()

    def build_model(self):
        inputs = layers.Input(shape=(self.win_length, self.channels), name="input_eeg")
        
        # Conv Block 1
        x = layers.Conv1D(filters=64, kernel_size=5, strides=2, padding='valid', name='conv1')(inputs)
        x = layers.LeakyReLU(alpha=0.2)(x) # تعديل alpha لتطابق TF1
        x = layers.MaxPooling1D(pool_size=2, strides=2)(x)

        # Conv Block 2
        x = layers.Conv1D(filters=64, kernel_size=3, strides=2, padding='valid', name='conv2')(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.MaxPooling1D(pool_size=2, strides=2)(x)

        # Conv Block 3
        x = layers.Conv1D(filters=64, kernel_size=3, strides=1, padding='valid', name='conv3')(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.MaxPooling1D(pool_size=2, strides=2)(x)

        # Conv Block 4
        x = layers.Conv1D(filters=128, kernel_size=2, strides=1, padding='valid', name='conv4')(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.MaxPooling1D(pool_size=2, strides=2)(x)

        # GRU Layer
        x = layers.GRU(128, name='gru_layer')(x)

        # Output logits (TF1-style); softmax is applied only for reporting/evaluation.
        outputs = layers.Dense(2, activation=None, name='dense_1')(x)

        model = Model(inputs=inputs, outputs=outputs, name="Epilepsy_Seizure_Model")
        return model

class Robust_CNN_GRU(CNN_GRU_Modern):
    def __init__(self, dim, noise_limit=0.3, l2_reg=0.001, lamda=1.0):
        super().__init__(dim, noise_limit, l2_reg)
        self.lamda = lamda
        # إضافة Gradient Clipping للـ Optimizer كما في TF1
        self.model_optimizer = optimizers.Adam(learning_rate=0.001, clipvalue=1.0)
        self.loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        self.noise_optimizer = optimizers.Adam(learning_rate=0.001)

    def generate_adversarial_noise(self, x_input, target_labels, steps=100, lr=0.001):
        """
        توليد ضجيج عدائي يحاكي التحسين التكراري في TF1
        """
        noise = tf.Variable(tf.zeros_like(x_input), trainable=True)
        self.noise_optimizer.learning_rate = lr
        
        for i in tf.range(steps):
            with tf.GradientTape() as tape:
                tape.watch(noise)
                adv_x = x_input + noise
                logits = self.model(adv_x, training=False)
                
                # حساب الخسارة: CrossEntropy + L2 loss للضجيج (كما في TF1)
                ce_loss = self.loss_fn(target_labels, logits)
                l2_ae_loss = self.lamda * tf.nn.l2_loss(noise)
                total_adv_loss = ce_loss + l2_ae_loss
            
            gradients = tape.gradient(total_adv_loss, noise)
            self.noise_optimizer.apply_gradients([(gradients, noise)])
            
            # تقييد الضجيج (Clipping) ضمن noise_limit
            noise.assign(tf.clip_by_value(noise, -self.noise_limit, self.noise_limit))
            
        return x_input + noise

    def _l2_all_trainables(self):
        if not self.model.trainable_variables:
            return tf.constant(0.0, dtype=tf.float32)
        return self.l2_reg * tf.add_n([tf.nn.l2_loss(v) for v in self.model.trainable_variables])

    def train_step(self, x_batch, y_batch):
        """خطوة تدريب عادية (بدون أمثلة عدائية)"""
        with tf.GradientTape() as tape:
            logits = self.model(x_batch, training=True)
            # TF1-style: CE + L2 على كل trainable vars
            loss = self.loss_fn(y_batch, logits) + self._l2_all_trainables()
            
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model_optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

    def train(self, x_train, y_train, x_val, y_val, epochs, batch_size, verbose=False):
        """
        محاكاة نمط التدريب في TF1:
        - batches متتالية بدون shuffle داخل epoch.
        - تحديث optimizer مرة إضافية على validation في نهاية كل epoch.
        """
        nb_batches = int(x_train.shape[0] / batch_size)
        if nb_batches <= 0:
            nb_batches = 1

        for epoch in range(epochs):
            batch_losses = []
            for batch in range(nb_batches):
                start = batch * batch_size
                end = min(start + batch_size, x_train.shape[0])
                xb = x_train[start:end]
                yb = y_train[start:end]
                loss = self.train_step(xb, yb)
                batch_losses.append(float(loss.numpy()))

            # TF1 behavior: تحديث مباشر على validation.
            _ = self.train_step(x_val, y_val)

            if verbose or ((epoch + 1) % 10 == 0):
                mean_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
                print(f"Epoch {epoch+1}/{epochs} - Loss: {mean_loss:.4f}")

    def _run_tf1_style_train(self, x_train, y_train, x_val, y_val, epochs, batch_size, verbose=False):
        """Alias للحفاظ على التوافق الخلفي مع الاستدعاءات القديمة في هذا الفرع."""
        return self.train(x_train, y_train, x_val, y_val, epochs, batch_size, verbose)

    def optimize_ae(self, actual, X, epochs, lr=0.001):
        """اسم متوافق مع TF1: تحسين مثال واحد نحو الهدف المعاكس."""
        return self.generate_adversarial_noise(X, actual, steps=epochs, lr=lr)

    def generate_ae(self, n, X, y, steps=100):
        """
        اسم متوافق مع TF1: توليد مجموعة AEs من بيانات (train+val).
        يعيد:
        - input_ae: أمثلة مولدة
        - y_ae: التسميات الأصلية للأمثلة المختارة
        """
        n = int(max(1, n))
        y_np = np.asarray(y)
        x_np = np.asarray(X)

        inter_count = int(n * (1.0 / 3.0))
        pre_count = n - inter_count
        state_pre = np.array([0, 1], dtype=np.float32)
        state_inter = np.array([1, 0], dtype=np.float32)

        data = []
        labels = []
        for i in range(n):
            state = state_pre if i < pre_count else state_inter
            target = np.flip(state).reshape(1, 2)

            idx = np.random.randint(0, y_np.shape[0])
            while not np.array_equal(y_np[idx], state):
                idx = np.random.randint(0, y_np.shape[0])

            inp = x_np[idx:idx+1]
            ae = self.optimize_ae(target, inp, steps)
            data.append(ae)
            labels.append(y_np[idx:idx+1])

            if (i + 1) % 25 == 0 or (i + 1) == n:
                print(f"Generated AE: {i+1}/{n}")

        input_ae = tf.concat(data, axis=0)
        y_ae = tf.concat(labels, axis=0)
        return input_ae, y_ae

    def train_with_adversarial(
        self,
        x_train,
        y_train,
        x_val=None,
        y_val=None,
        epochs_initial=50,
        epochs_adversarial=15,
        batch_size=32,
        percentage=0.1,
        micro_batch_size=8,
        adv_batch_size=2,
        adv_steps=10,
    ):
        """
        دالة التدريب الكاملة:
        1. تدريب أولي (Regular Training)
        2. توليد أمثلة عدائية (AE Generation)
        3. تدريب إضافي مع الأمثلة العدائية (AE Training)
        """
        if x_val is None or y_val is None or len(x_val) == 0:
            x_val = x_train
            y_val = y_train

        # 1. التدريب الأولي
        print(f"Starting Initial Training for {epochs_initial} epochs...")
        self.train(
            x_train, y_train,
            x_val, y_val,
            epochs=epochs_initial,
            batch_size=batch_size,
            verbose=False
        )

        # 2. توليد الأمثلة العدائية
        print(f"Generating Adversarial Examples ({percentage*100:.1f}% of data)...")
        ae_source_x = np.concatenate([np.asarray(x_train), np.asarray(x_val)], axis=0)
        ae_source_y = np.concatenate([np.asarray(y_train), np.asarray(y_val)], axis=0)
        num_ae = max(1, int(len(ae_source_x) * percentage))
        x_adv, y_adv_labels = self.generate_ae(num_ae, ae_source_x, ae_source_y, steps=adv_steps)

        # 3. التدريب الإضافي مع الأمثلة العدائية
        print(f"Starting Adversarial Training for {epochs_adversarial} epochs...")
        x_combined = tf.concat([x_train, x_adv], axis=0)
        y_combined = tf.concat([y_train, y_adv_labels], axis=0)

        # TF1: shuffle بعد ضم AEs ثم تدريب إضافي.
        idx = np.arange(x_combined.shape[0])
        np.random.shuffle(idx)
        x_combined = tf.gather(x_combined, idx)
        y_combined = tf.gather(y_combined, idx)

        self.train(
            x_combined, y_combined,
            x_val, y_val,
            epochs=epochs_adversarial,
            batch_size=batch_size,
            verbose=True
        )
        
        print("Training Complete.")

    def predict(self, x):
        """
        Return class probabilities (softmax) while training internally uses logits.
        """
        logits = self.model.predict(x)
        return tf.nn.softmax(logits, axis=-1).numpy()
