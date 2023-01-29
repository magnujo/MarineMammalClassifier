from MLModels import CNN_tensor as cnn
from Utils import FileUtils as filu
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Stores the values that maximizes mean fold score (the mean score of all the folds)
optimal_mean_values = {"best_mean_score": 0,
                       "optimal_batch_size": 0,
                       "optimal_learning_rate": 0}

# Stores the values that maximizes max fold score (the max score of all the folds)
optimal_max_values = {"best_fold_score": 0,
                      "optimal_batch_size": 0,
                      "optimal_learning_rate": 0,
                      "optimal_fold": 0}

history = {"learning_rates": {}}


#  Makes a dictionary that logs the different results so it can be plotted later
def update_history(max_score, mean_score, learning_rate, batch_size):
    if str(learning_rate) in history["learning_rates"]:
        history["learning_rates"][str(learning_rate)]["max_scores"].append(max_score)
        history["learning_rates"][str(learning_rate)]["mean_scores"].append(mean_score)
        history["learning_rates"][str(learning_rate)]["batch_size"].append(batch_size)

    else:
        history["learning_rates"][str(learning_rate)] = {"max_scores": [], "mean_scores": [], "batch_size": []}
        history["learning_rates"][str(learning_rate)]["max_scores"].append(max_score)
        history["learning_rates"][str(learning_rate)]["mean_scores"].append(mean_score)
        history["learning_rates"][str(learning_rate)]["batch_size"].append(batch_size)


def update_mean_scores(mean_score, batch_size, learning_rate):
    optimal_mean_values["best_mean_score"] = mean_score
    optimal_mean_values["optimal_batch_size"] = batch_size
    optimal_mean_values["optimal_learning_rate"] = learning_rate
    print(f"Batch size 1 and learning rate {learning_rate} got the new high score!")


def update_max_scores(max_score, best_fold, batch_size, learning_rate):
    optimal_max_values["best_fold_score"] = max_score
    optimal_max_values["optimal_batch_size"] = batch_size
    optimal_max_values["optimal_learning_rate"] = learning_rate
    optimal_max_values["optimal_fold"] = best_fold
    print(f"Batch size 1 and learning rate {learning_rate} got the new high score!")



def optimize_parameters(epochs, early_stop, folds, learning_rates, batch_sizes, model_names, machine,
                        specific_fold=None):
    if machine == "local":
        Y_PATH = r"C:\datasets\whoi\arrays\spectrograms\vgg_specs\labels.npy"
        MAPPINGS_PATH = r"C:\datasets\whoi\arrays\spectrograms\vgg_specs\mapping.json"
        X_PATH = r"C:\datasets\whoi\arrays\spectrograms\vgg_specs\spectrograms.npy"
        OUTPUT_DIR = r"G:\My Drive\WhaleClassifer\Results\WHOI"
    elif machine == "ucloud":
        X_PATH = "/work/Datasets_desktop/whoi/arrays/spectrograms/vgg_specs/spectrograms.npy"
        Y_PATH = "/work/Datasets_desktop/whoi/arrays/spectrograms/vgg_specs/labels.npy"
        MAPPINGS_PATH = "/work/Datasets_desktop/whoi/arrays/spectrograms/vgg_specs/mapping.json"
        OUTPUT_DIR = "/work/Outputs"
    elif machine == "colab":
        Y_PATH = "/content/drive/Othercomputers/My_Computer/whoi/arrays/spectrograms/vgg_specs/labels.npy"
        MAPPINGS_PATH = "/content/drive/Othercomputers/My_Computer/whoi/arrays/spectrograms/vgg_specs/mapping.json"
        X_PATH = "/content/drive/Othercomputers/My_Computer/whoi/arrays/spectrograms/vgg_specs/spectrograms.npy"
        OUTPUT_DIR = "/content/drive/Othercomputers/My_Computer/WhaleFM/Results/WHOI"
    else:
        raise ValueError(f"Got invalid machine input: {machine}. Accepted values are: local, ucloud or colab")

    results_dir = os.path.join(OUTPUT_DIR, str(len(os.listdir(OUTPUT_DIR)) + 1))
    os.mkdir(results_dir)

    print("Finding optimal model using following parameters:")
    print(
        f" Epochs: {epochs} \n early_stop: {early_stop} \n folds: {folds} \n learning_rates: {learning_rates} \n batch_sizes: {batch_sizes} \n")

    global optimal_mean_values
    global optimal_max_values
    global history

    for model_name in model_names:
        optimal_model = None
        mapping = None
        best_test_dataset = None
        best_x = None
        best_y = None
        optimal_mean_values = {"best_mean_score": 0,
                               "optimal_batch_size": 0,
                               "optimal_learning_rate": 0}
        # Stores the values that maximizes max fold score (the max score of all the folds)
        optimal_max_values = {"best_fold_score": 0,
                              "optimal_batch_size": 0,
                              "optimal_learning_rate": 0,
                              "optimal_fold": 0}
        history = {"learning_rates": {}}

        print(f"Finding optimal values for model: {model_name}")
        for batch_size in batch_sizes:
            for learning_rate in learning_rates:
                print(f"Training with batch size = {batch_size} and learning_rate = {learning_rate}")
                mapping, mean_score, best_fold, best_model, test_dataset, max_fold_score, x, y = cnn.run(
                    mapping_json=MAPPINGS_PATH,
                    labels_npy=Y_PATH,
                    features_npy=X_PATH,
                    batch_size=batch_size,
                    epochs=epochs,
                    learning_rate=learning_rate,
                    model_name=model_name,
                    error_function="sparse_categorical_crossentropy",
                    early_stop=early_stop, folds=folds, specific_fold=specific_fold)

                print(f"Results: \n mean_score: {mean_score} \n max_score: {max_fold_score} \n best_fold: {best_fold}")

                update_history(max_fold_score, mean_score, learning_rate, batch_size)

                if mean_score > optimal_mean_values["best_mean_score"]:
                    update_mean_scores(mean_score=mean_score, batch_size=batch_size,
                                       learning_rate=learning_rate)
                    mapping = mapping
                    best_x = x
                    best_y = y
                    best_test_dataset = test_dataset

                if max_fold_score > optimal_max_values["best_fold_score"]:
                    update_max_scores(max_score=max_fold_score, best_fold=best_fold,
                                      batch_size=batch_size, learning_rate=learning_rate)
                    mapping = mapping
                    optimal_model = best_model

        print(f"Optimizing done.")
        model_dir = os.path.join(results_dir, model_name)
        os.mkdir(model_dir)
        filu.save_model_to_h5(optimal_model, os.path.join(model_dir, "WAV_SPEC_CNN.h5"))
        with open(os.path.join(model_dir, 'mapping.json'), 'w') as f:
            json.dump(mapping, f)
        with open(os.path.join(model_dir, 'optimal_mean_values.json'), 'w') as f:
            json.dump(optimal_mean_values, f)
        with open(os.path.join(model_dir, 'optimal_max_values.json'), 'w') as f:
            json.dump(optimal_max_values, f)
        with open(os.path.join(model_dir, 'history.json'), 'w') as f:
            json.dump(history, f)
        np.save(os.path.join(model_dir, 'x.npy'), best_x)
        np.save(os.path.join(model_dir, 'y.npy'), best_y)
        tf.data.experimental.save(best_test_dataset, os.path.join(model_dir, 'test_dataset'))


def display_history_plot(history, metric):
    if metric == "acc":
        val_acc = history.history["val_accuracy"]
        train_acc = history.history["accuracy"]

        fig, ax = plt.subplots()
        ax.plot(val_acc, 'r--', label='val_accuracy')
        ax.plot(train_acc, 'k', label='train_accuracy')
        legend = ax.legend()
        plt.show()

    elif metric == "loss":
        val_acc = history.history["val_loss"]
        train_acc = history.history["loss"]

        fig, ax = plt.subplots()
        ax.plot(val_acc, 'r--', label='val_loss')
        ax.plot(train_acc, 'k', label='train_loss')
        legend = ax.legend()
        plt.show()

    else:
        raise ValueError("Metric value not supported")