import os
import time
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

from models.helping_functions import calc_metrics, next_batch, shuffle_data


class Robust_CNN_GRU:
    """
    TF1-style execution on TF2 runtime via tf.compat.v1:
    - static graph + placeholders
    - Session.run training/evaluation
    - Saver checkpoints
    """

    def __init__(self, dim, dataset="CHBMIT", noise_limit=0.3, l2_reg=0.001, lamda=1.0):
        self.win_length = dim[0]
        self.channels = dim[1]
        self.dataset = dataset
        self.noise_limit = noise_limit
        self.l2_reg = l2_reg
        self.lamda_default = lamda

        self.graph = tf.Graph()
        with self.graph.as_default():
            self._build_graph()
            self.session = tf.compat.v1.Session(graph=self.graph)
            self.saver = tf.compat.v1.train.Saver()
            self.session.run(tf.compat.v1.global_variables_initializer())
            self.init_noise()

    def _build_graph(self):
        self.input = tf.compat.v1.placeholder(
            tf.float32, [None, self.win_length, self.channels], name="input"
        )
        self.Y = tf.compat.v1.placeholder(tf.float32, [None, 2], name="Y")
        self.cnn_keep_rate = tf.compat.v1.placeholder_with_default(1.0, shape=(), name="cnn_keep_rate")
        self.gru_keep_rate = tf.compat.v1.placeholder_with_default(1.0, shape=(), name="gru_keep_rate")
        self.lamda = tf.compat.v1.placeholder_with_default(self.lamda_default, shape=(), name="lambda")

        self.ADVERSARY_VARIABLES = "adversary_variables"
        collections = [tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, self.ADVERSARY_VARIABLES]
        self.x_noise = tf.compat.v1.Variable(
            tf.zeros([1, self.win_length, self.channels], dtype=tf.float32),
            name="x_noise",
            trainable=False,
            collections=collections,
        )
        self.x_ae2 = self.input + self.x_noise
        self.x_noise_clip = tf.compat.v1.assign(
            self.x_noise,
            tf.clip_by_value(self.x_noise, -self.noise_limit, self.noise_limit),
        )

        with tf.compat.v1.variable_scope("conv_1"):
            conv1 = tf.keras.layers.Conv1D(
                filters=64, kernel_size=5, strides=2, padding="valid", activation=tf.nn.leaky_relu, name="conv1"
            )(self.x_ae2)
            max_pool_1 = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2)(conv1)

        with tf.compat.v1.variable_scope("conv_2"):
            conv2 = tf.keras.layers.Conv1D(
                filters=64, kernel_size=3, strides=2, padding="valid", activation=tf.nn.leaky_relu, name="conv2"
            )(max_pool_1)
            max_pool_2 = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2)(conv2)

        with tf.compat.v1.variable_scope("conv_3"):
            conv3 = tf.keras.layers.Conv1D(
                filters=64, kernel_size=3, strides=1, padding="valid", activation=tf.nn.leaky_relu, name="conv3"
            )(max_pool_2)
            max_pool_3 = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2)(conv3)

        with tf.compat.v1.variable_scope("conv_4"):
            conv4 = tf.keras.layers.Conv1D(
                filters=128, kernel_size=2, strides=1, padding="valid", activation=tf.nn.leaky_relu, name="conv4"
            )(max_pool_3)
            max_pool_4 = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2)(conv4)

        with tf.compat.v1.variable_scope("GRU"):
            self.embeddings1 = tf.keras.layers.GRU(128, name="gru_layer")(max_pool_4)
            output = tf.keras.layers.Dense(2, activation=None, name="dense_1")(self.embeddings1)

        self.cost_Y = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=self.Y)
        )
        l2 = self.l2_reg * tf.add_n([tf.nn.l2_loss(v) for v in tf.compat.v1.trainable_variables()])
        self.total_loss = self.cost_Y + l2

        optimizer = tf.compat.v1.train.AdamOptimizer()
        gvs = optimizer.compute_gradients(self.total_loss)
        capped_gvs = []
        for grad, var in gvs:
            if grad is None:
                capped_gvs.append((grad, var))
            else:
                capped_gvs.append((tf.clip_by_value(grad, -1.0, 1.0), var))
        self.Y_op = optimizer.apply_gradients(capped_gvs)

        adversary_variables = tf.compat.v1.get_collection(self.ADVERSARY_VARIABLES)
        l2_ae = self.lamda * tf.nn.l2_loss(self.x_noise, name="L2")
        self.loss_adversary = self.cost_Y + l2_ae
        self.optimizer_ae = tf.compat.v1.train.AdamOptimizer().minimize(
            self.loss_adversary, var_list=adversary_variables, name="adversarial_optimizer"
        )
        self.predictions = tf.nn.softmax(output, name="Preds")

    def reset_variables(self):
        with self.graph.as_default():
            self.session.run(tf.compat.v1.global_variables_initializer())
            self.init_noise()

    def init_noise(self):
        self.session.run(tf.compat.v1.variables_initializer([self.x_noise]))

    def optimize_ae(self, actual, X, epochs):
        self.init_noise()
        X2 = X
        for _ in range(epochs):
            feed_dict = {self.Y: actual, self.input: X2, self.lamda: 0.001}
            ae, _, _ = self.session.run([self.x_ae2, self.optimizer_ae, self.loss_adversary], feed_dict=feed_dict)
            self.session.run(self.x_noise_clip)
            X2 = ae
        return X2

    def generate_ae(self, n, X, y, steps=100):
        n = int(max(1, n))
        inter_count = int(n * (1.0 / 3.0))
        pre_count = n - inter_count

        preictal = np.array([0, 1])
        interictal = np.array([1, 0])
        data = []
        labels = []

        for i in range(n):
            state = preictal if i < pre_count else interictal
            target = np.flip(state).reshape(-1, 2)
            idx = np.random.randint(0, y.shape[0])
            while (y[idx] != state).all():
                idx = np.random.randint(0, y.shape[0])
            inp = X[idx, :, :].reshape(-1, X.shape[1], X.shape[2])
            ae = self.optimize_ae(target, inp, steps)
            data.append(ae)
            labels.append(y[idx].reshape(-1, 2))
            if (i + 1) % 25 == 0 or (i + 1) == n:
                print(f"Generated AE: {i + 1}/{n}")

        input_ae = np.concatenate(data, axis=0)
        y_ae = np.concatenate(labels, axis=0)
        return input_ae, y_ae

    @staticmethod
    def _ckpt_exists(prefix: str) -> bool:
        if not prefix:
            return False
        # TF Saver writes multiple files; ".index" is a stable indicator.
        return os.path.isfile(prefix + ".index")

    @staticmethod
    def _ae_cache_paths(ae_cache_dir: str):
        return (
            os.path.join(ae_cache_dir, "input_ae.npy"),
            os.path.join(ae_cache_dir, "labels_ae.npy"),
            os.path.join(ae_cache_dir, "meta.json"),
        )

    @classmethod
    def _try_load_ae_cache(cls, ae_cache_dir: str, expected_meta: dict):
        if not ae_cache_dir:
            return None
        input_path, labels_path, meta_path = cls._ae_cache_paths(ae_cache_dir)
        if not (os.path.isfile(input_path) and os.path.isfile(labels_path) and os.path.isfile(meta_path)):
            return None
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            return None

        # Validate only the key bits that must match for safe reuse.
        for k in ("n_ae", "adv_steps", "percentage", "win_length", "channels"):
            if k in expected_meta and meta.get(k) != expected_meta.get(k):
                return None

        try:
            input_ae = np.load(input_path, allow_pickle=False)
            labels_ae = np.load(labels_path, allow_pickle=False)
        except Exception:
            return None

        if input_ae.ndim != 3 or labels_ae.ndim != 2:
            return None
        if input_ae.shape[0] != labels_ae.shape[0]:
            return None
        if input_ae.shape[1] != expected_meta.get("win_length") or input_ae.shape[2] != expected_meta.get("channels"):
            return None
        if labels_ae.shape[1] != 2:
            return None
        return input_ae, labels_ae, meta

    @classmethod
    def _save_ae_cache(cls, ae_cache_dir: str, input_ae: np.ndarray, labels_ae: np.ndarray, meta: dict):
        if not ae_cache_dir:
            return
        os.makedirs(ae_cache_dir, exist_ok=True)
        input_path, labels_path, meta_path = cls._ae_cache_paths(ae_cache_dir)
        np.save(input_path, input_ae)
        np.save(labels_path, labels_ae)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=True, indent=2, sort_keys=True)

    def train(self, x_train, y_train, x_val, y_val, epochs, batch_size, verbose=False):
        nb_batches = int(x_train.shape[0] / batch_size)
        if nb_batches <= 0:
            nb_batches = 1

        start_time = time.time()
        for _epoch in range(epochs):
            for batch in range(nb_batches):
                xb, yb = next_batch(x_train, y_train, batch, batch_size)
                feed = {
                    self.input: xb,
                    self.Y: yb,
                    self.cnn_keep_rate: 1.0,
                    self.gru_keep_rate: 1.0,
                }
                self.session.run(self.Y_op, feed_dict=feed)

            val_feed = {
                self.input: x_val,
                self.Y: y_val,
                self.cnn_keep_rate: 1.0,
                self.gru_keep_rate: 1.0,
            }
            self.session.run(self.Y_op, feed_dict=val_feed)
            preds_val, y_loss = self.session.run([self.predictions, self.cost_Y], feed_dict=val_feed)

            if verbose:
                auc, acc, sens, _ = calc_metrics(y_val, preds_val)
                elapsed = time.time() - start_time
                print(
                    f"sensitivity: {round(sens, 2)} AUC: {round(auc, 2)} "
                    f"Y_loss: {round(float(y_loss), 2)} acc: {round(acc, 2)} time: {round(elapsed, 2)}"
                )

    def train_with_adversarial(
        self,
        x_train,
        y_train,
        x_val=None,
        y_val=None,
        epochs_initial=50,
        epochs_adversarial=15,
        batch_size=256,
        percentage=0.4,
        micro_batch_size=None,
        adv_batch_size=None,
        adv_steps=100,
        ae_cache_dir=None,
        resume=False,
        initial_ckpt_prefix=None,
    ):
        if x_val is None or y_val is None or len(x_val) == 0:
            x_val = x_train
            y_val = y_train

        X = np.concatenate((x_train, x_val), axis=0)
        y = np.concatenate((y_train, y_val), axis=0)
        n_ae = int(X.shape[0] * percentage)

        expected_meta = dict(
            n_ae=int(n_ae),
            adv_steps=int(adv_steps),
            percentage=float(percentage),
            win_length=int(X.shape[1]),
            channels=int(X.shape[2]),
        )

        cache = self._try_load_ae_cache(ae_cache_dir, expected_meta)
        can_resume = bool(resume) and cache is not None and self._ckpt_exists(initial_ckpt_prefix)
        if can_resume:
            print("Resuming adversarial phase from cached AEs + initial checkpoint...")
            self.saver.restore(sess=self.session, save_path=initial_ckpt_prefix)
            input_ae, labels_ae, _meta = cache
        else:
            self.train(x_train, y_train, x_val, y_val, epochs_initial, batch_size, verbose=False)
            if initial_ckpt_prefix:
                self.save_weights(initial_ckpt_prefix)

            print(f"Number of AEs to generate: {n_ae}")
            input_ae, labels_ae = self.generate_ae(n_ae, X, y, steps=adv_steps)
            if ae_cache_dir:
                meta = dict(
                    **expected_meta,
                    created_unix=int(time.time()),
                    input_shape=list(input_ae.shape),
                    labels_shape=list(labels_ae.shape),
                )
                self._save_ae_cache(ae_cache_dir, input_ae, labels_ae, meta)

        x_train_aug = np.concatenate((x_train, input_ae), axis=0)
        y_train_aug = np.concatenate((y_train, labels_ae), axis=0)
        x_train_aug, y_train_aug = shuffle_data(x_train_aug, y_train_aug)

        self.train(x_train_aug, y_train_aug, x_val, y_val, epochs_adversarial, batch_size, verbose=True)

    def testing(self, x_test, y_test):
        feed = {
            self.input: x_test,
            self.Y: y_test,
            self.cnn_keep_rate: 1.0,
            self.gru_keep_rate: 1.0,
        }
        test_loss, preds, emb = self.session.run([self.cost_Y, self.predictions, self.embeddings1], feed_dict=feed)
        auc_test, acc, sensitivity, false_alarm = calc_metrics(y_test, preds)
        return auc_test, acc, sensitivity, false_alarm, float(test_loss), emb, preds

    def predict(self, x):
        return self.session.run(self.predictions, feed_dict={self.input: x})

    def save_weights(self, save_path):
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        self.saver.save(sess=self.session, save_path=save_path)

    def close(self):
        try:
            self.session.close()
        except Exception:
            pass
