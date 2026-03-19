import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, regularizers

class CNN_GRU_Modern:
    def __init__(self, dim, noise_limit=0.02, l2_reg=0.001):
        self.win_length = dim[0]
        self.channels = dim[1]
        self.noise_limit = noise_limit
        self.l2_reg = l2_reg # إضافة L2 Regularization
        self.model = self.build_model()

    def build_model(self):
        inputs = layers.Input(shape=(self.win_length, self.channels), name="input_eeg")
        
        # استخدام regularizer
        reg = regularizers.l2(self.l2_reg)

        # Conv Block 1
        x = layers.Conv1D(filters=64, kernel_size=5, strides=2, padding='valid', 
                          kernel_regularizer=reg, name='conv1')(inputs)
        x = layers.LeakyReLU(alpha=0.2)(x) # تعديل alpha لتطابق TF1
        x = layers.MaxPooling1D(pool_size=2, strides=2)(x)

        # Conv Block 2
        x = layers.Conv1D(filters=64, kernel_size=3, strides=2, padding='valid', 
                          kernel_regularizer=reg, name='conv2')(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.MaxPooling1D(pool_size=2, strides=2)(x)

        # Conv Block 3
        x = layers.Conv1D(filters=64, kernel_size=3, strides=1, padding='valid', 
                          kernel_regularizer=reg, name='conv3')(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.MaxPooling1D(pool_size=2, strides=2)(x)

        # Conv Block 4
        x = layers.Conv1D(filters=128, kernel_size=2, strides=1, padding='valid', 
                          kernel_regularizer=reg, name='conv4')(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.MaxPooling1D(pool_size=2, strides=2)(x)

        # GRU Layer
        x = layers.GRU(128, kernel_regularizer=reg, name='gru_layer')(x)

        # Output Layer
        outputs = layers.Dense(2, activation='softmax', kernel_regularizer=reg, name='predictions')(x)

        model = Model(inputs=inputs, outputs=outputs, name="Epilepsy_Seizure_Model")
        return model

class Robust_CNN_GRU(CNN_GRU_Modern):
    def __init__(self, dim, noise_limit=0.3, l2_reg=0.001, lamda=1.0):
        super().__init__(dim, noise_limit, l2_reg)
        self.lamda = lamda
        # إضافة Gradient Clipping للـ Optimizer كما في TF1
        self.model_optimizer = optimizers.Adam(learning_rate=0.001, clipvalue=1.0)
        self.loss_fn = tf.keras.losses.CategoricalCrossentropy()

    @tf.function
    def generate_adversarial_noise(self, x_input, target_labels, steps=100, lr=0.001):
        """
        توليد ضجيج عدائي يحاكي التحسين التكراري في TF1
        """
        noise = tf.zeros_like(x_input)
        
        for i in tf.range(steps):
            with tf.GradientTape() as tape:
                tape.watch(noise)
                adv_x = x_input + noise
                predictions = self.model(adv_x, training=False)
                
                # حساب الخسارة: CrossEntropy + L2 loss للضجيج (كما في TF1)
                ce_loss = self.loss_fn(target_labels, predictions)
                l2_ae_loss = self.lamda * tf.nn.l2_loss(noise)
                total_adv_loss = ce_loss + l2_ae_loss
            
            # حساب التدرج بالنسبة للضجيج
            gradients = tape.gradient(total_adv_loss, noise)
            
            # تحديث الضجيج (Gradient Descent لتقليل الخسارة تجاه الهدف المعاكس)
            noise.assign_sub(lr * gradients)
            
            # تقييد الضجيج (Clipping) ضمن noise_limit
            noise.assign(tf.clip_by_value(noise, -self.noise_limit, self.noise_limit))
            
        return x_input + noise

    @tf.function
    def train_step(self, x_batch, y_batch):
        """خطوة تدريب عادية (بدون أمثلة عدائية)"""
        with tf.GradientTape() as tape:
            predictions = self.model(x_batch, training=True)
            # الخسارة تشمل CrossEntropy + L2 Regularization losses
            loss = self.loss_fn(y_batch, predictions) + sum(self.model.losses)
            
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model_optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

    def train_with_adversarial(self, x_train, y_train, epochs_initial=50, epochs_adversarial=15, batch_size=256, percentage=0.4):
        """
        دالة التدريب الكاملة:
        1. تدريب أولي (Regular Training)
        2. توليد أمثلة عدائية (AE Generation)
        3. تدريب إضافي مع الأمثلة العدائية (AE Training)
        """
        # 1. التدريب الأولي
        print(f"Starting Initial Training for {epochs_initial} epochs...")
        train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(len(x_train)).batch(batch_size)
        
        for epoch in range(epochs_initial):
            epoch_loss = []
            for x_batch, y_batch in train_ds:
                loss = self.train_step(x_batch, y_batch)
                epoch_loss.append(loss)
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs_initial} - Loss: {np.mean(epoch_loss):.4f}")

        # 2. توليد الأمثلة العدائية
        print(f"Generating Adversarial Examples ({percentage*100}% of data)...")
        num_ae = int(len(x_train) * percentage)
        indices = np.random.choice(len(x_train), num_ae, replace=False)
        x_sample = tf.gather(x_train, indices)
        y_sample = tf.gather(y_train, indices)
        
        # الهدف المعاكس (Target): قلب الـ labels
        # [1, 0] (Interictal) -> [0, 1] (Preictal)
        # [0, 1] (Preictal) -> [1, 0] (Interictal)
        target_labels = tf.reverse(y_sample, axis=[-1])
        
        # توليد الضجيج
        x_adv = self.generate_adversarial_noise(x_sample, target_labels, steps=100)
        
        # 3. التدريب الإضافي مع الأمثلة العدائية
        print(f"Starting Adversarial Training for {epochs_adversarial} epochs...")
        x_combined = tf.concat([x_train, x_adv], axis=0)
        y_combined = tf.concat([y_train, y_sample], axis=0) # نستخدم الـ labels الأصلية
        
        combined_ds = tf.data.Dataset.from_tensor_slices((x_combined, y_combined)).shuffle(len(x_combined)).batch(batch_size)
        
        for epoch in range(epochs_adversarial):
            epoch_loss = []
            for x_batch, y_batch in combined_ds:
                loss = self.train_step(x_batch, y_batch)
                epoch_loss.append(loss)
            print(f"Adv Epoch {epoch+1}/{epochs_adversarial} - Loss: {np.mean(epoch_loss):.4f}")
        
        print("Training Complete.")