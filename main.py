import os
import sys
import argparse
import warnings
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

from utils.load_signals import PrepData
from models.helping_functions import train_val_cv_split, train_val_test_split, collect_results
from utils.load_results import summary_results, load_results
from utils.save_load import savefile
from models.model import Robust_CNN_GRU

if not sys.warnoptions:
    warnings.simplefilter("ignore")


def get_args():
    parser = argparse.ArgumentParser(description="Robust Seizure Prediction (TF1-style execution on TF2 runtime)")
    parser.add_argument("-d", "--dataset", type=str, default="CHBMIT", help="Dataset name")
    parser.add_argument("-p_id", "--patient_id", type=str, default="all", help="Patient ID or all")
    parser.add_argument("-val", "--validation", type=str, default="cv", choices=["cv", "test"], help="Validation mode")
    parser.add_argument("-m", "--mode", type=str, default="AE", choices=["AE", "without_AE"], help="Training mode")
    parser.add_argument("-b", "--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("-e", "--epoch", type=int, default=50, help="Initial training epochs")
    parser.add_argument("-e_adv", "--epochs_adversarial", type=int, default=15, help="Adversarial training epochs")
    parser.add_argument("--adv_steps", type=int, default=100, help="Adversarial optimization steps")
    parser.add_argument("-perc", "--percentage", type=float, default=40, help="AE percentage (40 or 0.4)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose training logs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--rebuild_cache", action="store_true", help="Force rebuilding dataset cache")
    parser.add_argument("--resume", action="store_true", help="Resume fold adversarial phase from cache/checkpoint if available")
    parser.add_argument("--no_ae_cache", action="store_true", help="Disable saving/loading adversarial example cache")
    return parser.parse_args()


def configure_gpu_memory_growth():
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass


def normalize_percentage(value):
    return float(value / 100.0) if float(value) > 1.0 else float(value)


def data_loading(target, settings):
    print("Data Loading...................................")
    ictal_X, ictal_y = PrepData(target, type="ictal", settings=settings).apply()
    interictal_X, interictal_y = PrepData(target, type="interictal", settings=settings).apply()
    return ictal_X, ictal_y, interictal_X, interictal_y


def _get_patients(dataset, patient_id):
    if str(patient_id).lower() != "all":
        return [str(patient_id)]
    if dataset == "CHBMIT":
        return ["1", "2", "3", "5", "9", "10", "13", "14", "18", "19", "20", "21", "23"]
    if dataset == "FB":
        return ["1", "3", "4", "5", "6", "14", "15", "16", "17", "18", "19", "20", "21"]
    return [str(patient_id)]


def _results_dir(dataset, mode, validation):
    if validation == "cv" and mode == "AE":
        return f"results/resultsCV_{dataset}_AE/"
    if validation == "cv":
        return f"results/results_CV_{dataset}/"
    if validation == "test" and mode == "AE":
        return f"results/results_{dataset}_AE/"
    return f"results/results_{dataset}/"


def train(args):
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    data_dir = r"E:\Graduation project\DATA\chb-mit-scalp-eeg-database-1.0.0\chb-mit-scalp-eeg-database-1.0.0"
    metadata_dir = "E:/Graduation project/Robust-Seizure-Prediction/data_configs"
    results_dir = _results_dir(args.dataset, args.mode, args.validation)
    os.makedirs(results_dir, exist_ok=True)

    settings = {
        "dataset": args.dataset,
        "datadir": data_dir,
        "metadata_dir": metadata_dir,
        "rebuild_cache": args.rebuild_cache,
    }

    patients = _get_patients(args.dataset, args.patient_id)
    percentage = normalize_percentage(args.percentage)
    ae_cache_enabled = not bool(args.no_ae_cache)

    for target in patients:
        print(f"\n--- Starting Training for Patient {target} ---")
        ictal_X, ictal_y, interictal_X, interictal_y = data_loading(target, settings)
        if len(ictal_X) < 2 or len(interictal_X) < 2:
            print(f"Insufficient folds for patient {target}. Skipping...")
            continue

        history = dict(
            AUC_AVG=[],
            y_pred=[],
            y_test=[],
            test_loss=[],
            test_sensitivity=[],
            test_false_alarm=[],
            embed1=[],
        )
        auc_score_total = 0.0
        fold_count = 0

        save_dir = f"checkpoints_{args.dataset}/pat{target}"
        os.makedirs(save_dir, exist_ok=True)

        if args.validation == "cv":
            folds = train_val_cv_split(ictal_X, ictal_y, interictal_X, interictal_y, 0.1, is_shuffeling=True)
            for fold_idx, (X_train, y_train_raw, X_val, y_val_raw, X_test, y_test_raw) in enumerate(folds, start=1):
                print(f"\n--- Fold {fold_idx} ---")
                X_train = np.asarray(X_train, dtype=np.float32)
                y_train = to_categorical(np.asarray(y_train_raw, dtype=np.int32))
                X_val = np.asarray(X_val, dtype=np.float32)
                y_val = to_categorical(np.asarray(y_val_raw, dtype=np.int32))
                X_test = np.asarray(X_test, dtype=np.float32)
                y_test = to_categorical(np.asarray(y_test_raw, dtype=np.int32))

                fold_dir = os.path.join(save_dir, f"fold{fold_idx:02d}")
                os.makedirs(fold_dir, exist_ok=True)
                save_path = os.path.join(fold_dir, f"CNN_GRU_{target}_fold{fold_idx:02d}")
                initial_ckpt_prefix = os.path.join(fold_dir, f"CNN_GRU_{target}_fold{fold_idx:02d}_initial")
                ae_cache_dir = os.path.join(fold_dir, "ae_cache") if ae_cache_enabled else None

                input_dim = (X_train.shape[1], X_train.shape[2])
                model_wrapper = Robust_CNN_GRU(dim=input_dim, dataset=args.dataset, noise_limit=0.3, l2_reg=0.001)
                model_wrapper.reset_variables()

                if args.mode == "AE":
                    model_wrapper.train_with_adversarial(
                        X_train,
                        y_train,
                        x_val=X_val,
                        y_val=y_val,
                        epochs_initial=args.epoch,
                        epochs_adversarial=args.epochs_adversarial,
                        batch_size=args.batch_size,
                        percentage=percentage,
                        adv_steps=args.adv_steps,
                        ae_cache_dir=ae_cache_dir,
                        resume=args.resume,
                        initial_ckpt_prefix=initial_ckpt_prefix,
                    )
                else:
                    model_wrapper.train(
                        X_train,
                        y_train,
                        X_val,
                        y_val,
                        epochs=args.epoch,
                        batch_size=args.batch_size,
                        verbose=args.verbose,
                    )

                auc_test, acc, sensitivity, false_alarm, test_loss1, emb, preds = model_wrapper.testing(X_test, y_test)
                auc_score_total += auc_test
                fold_count += 1
                history = collect_results(history, sensitivity, false_alarm, preds, y_test, test_loss1, emb)
                model_wrapper.save_weights(save_path)
                model_wrapper.close()
                print(
                    f"Fold {fold_idx} Results - Sensitivity: {sensitivity * 100:.2f}%, "
                    f"FPR: {false_alarm:.4f}, AUC: {auc_test:.4f}"
                )

            if fold_count > 0:
                history["AUC_AVG"].append(auc_score_total / fold_count)

        elif args.validation == "test":
            X_train, y_train_raw, X_val, y_val_raw, X_test, y_test_raw = train_val_test_split(
                ictal_X, ictal_y, interictal_X, interictal_y, 0.1, 0.2, is_shuffeling=True
            )
            X_train = np.asarray(X_train, dtype=np.float32)
            y_train = to_categorical(np.asarray(y_train_raw, dtype=np.int32))
            X_val = np.asarray(X_val, dtype=np.float32)
            y_val = to_categorical(np.asarray(y_val_raw, dtype=np.int32))
            X_test = np.asarray(X_test, dtype=np.float32)
            y_test = to_categorical(np.asarray(y_test_raw, dtype=np.int32))

            fold_dir = os.path.join(save_dir, "test")
            os.makedirs(fold_dir, exist_ok=True)
            save_path = os.path.join(fold_dir, f"CNN_GRU_{target}_test")
            initial_ckpt_prefix = os.path.join(fold_dir, f"CNN_GRU_{target}_test_initial")
            ae_cache_dir = os.path.join(fold_dir, "ae_cache") if ae_cache_enabled else None

            input_dim = (X_train.shape[1], X_train.shape[2])
            model_wrapper = Robust_CNN_GRU(dim=input_dim, dataset=args.dataset, noise_limit=0.3, l2_reg=0.001)
            model_wrapper.reset_variables()

            if args.mode == "AE":
                model_wrapper.train_with_adversarial(
                    X_train,
                    y_train,
                    x_val=X_val,
                    y_val=y_val,
                    epochs_initial=args.epoch,
                    epochs_adversarial=args.epochs_adversarial,
                    batch_size=args.batch_size,
                    percentage=percentage,
                    adv_steps=args.adv_steps,
                    ae_cache_dir=ae_cache_dir,
                    resume=args.resume,
                    initial_ckpt_prefix=initial_ckpt_prefix,
                )
            else:
                model_wrapper.train(
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    epochs=args.epoch,
                    batch_size=args.batch_size,
                    verbose=args.verbose,
                )

            auc_test, acc, sensitivity, false_alarm, test_loss1, emb, preds = model_wrapper.testing(X_test, y_test)
            auc_score_total += auc_test
            fold_count += 1
            history = collect_results(history, sensitivity, false_alarm, preds, y_test, test_loss1, emb)
            history["AUC_AVG"].append(auc_score_total / fold_count)
            model_wrapper.save_weights(save_path)
            model_wrapper.close()

        history_path = os.path.join(results_dir, f"history_{target}")
        savefile(history, history_path)
        print(f"Results for Patient {target} saved to {history_path}.pkl")

    return settings


def main():
    configure_gpu_memory_growth()
    tf.compat.v1.disable_eager_execution()
    args = get_args()
    settings = train(args)
    results_path = _results_dir(args.dataset, args.mode, args.validation)
    data_results, patients = load_results(results_path, args.dataset)
    summary_results(patients, data_results)


if __name__ == "__main__":
    main()
