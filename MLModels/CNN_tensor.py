# This file is based on code from https://www.youtube.com/watch?v=dOG-HxpbMSw&list=PL-wATfeyAMNrtbkCNsLcpoAyBBRJZVlnf&index=16&ab_channel=ValerioVelardo-TheSoundofAI
import numpy as np

#np.random.seed(42)
import tensorflow as tf
#tf.random.set_seed(42)
#tf.keras.utils.set_random_seed(42)

import json
from Utils import FileUtils as filu
from sklearn.metrics import ConfusionMatrixDisplay, roc_auc_score, matthews_corrcoef, cohen_kappa_score, \
    classification_report, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from tensorflow import keras
from MLModels import architectures
import os



def load_data(mapping_json, labels, features):
    with open(mapping_json, "r") as fp:
        mapping = json.load(fp)

    inputs = np.load(features)
    labels = np.load(labels)

    return inputs, labels, mapping


def cross_val(inputs, labels, test_size, folds, rgb=True):
    # Add a new dimension to the input so each input has the shape (224, 224, 1) (grayscale) or (244, 244, 3) (like a RGB image)
    if rgb:
        inputs = np.repeat(inputs[..., np.newaxis], 3,
                           -1)  # shape output: (num_samples, num_freq_bins, num_time_bins, 3)

    else:
        inputs = inputs[..., np.newaxis]  # shape result: (num_samples, num_freq_bins, num_time_bins, 1)

    x_train, x_test, y_train, y_test = train_test_split(inputs, labels,
                                                        test_size=test_size,
                                                        shuffle=True,
                                                        random_state=42,  # To make consistent testing/comparison
                                                        stratify=labels)

    skf = StratifiedKFold(n_splits=folds,
                          shuffle=True, random_state=42)  # to make consistent testing/comparison

    return skf, x_train, x_test, y_train, y_test


def val_train_split(inputs, labels, val_size, rgb):
    if rgb:
        inputs = np.repeat(inputs[..., np.newaxis], 3,
                           -1)  # shape output: (num_samples, num_freq_bins, num_time_bins, 3)

    else:
        inputs = inputs[..., np.newaxis]

    # Further split the train data into val and train data:
    x_train, x_val, y_train, y_val = train_test_split(inputs,
                                                      labels,
                                                      test_size=val_size,
                                                      shuffle=True)

    return x_train, x_val, y_train, y_val


def preprocess(x, model_name):
    f = getattr(keras.applications, model_name)
    x = f.preprocess_input(x)
    return x


def run(mapping_json,
        labels_npy,
        features_npy,
        batch_size,
        epochs,
        learning_rate, model_name,
        error_function="sparse_categorical_crossentropy",
        early_stop=True,
        folds=3, specific_fold=None, num_classes=None, prepocess=False,
        sub_set=None):
    """

    Runs multiple cross validations and return the mappings and a mean score of all the fold scores.

    :param mapping_json: Path to the json file that contains the mapping dictionary
    :param labels_npy: Path to the labels file that contains all the label integers. Expects a single .npy
    :param features_npy: Path to the file containing the features. Expects a single .npy file
    :param batch_size: Specifies how many inputs the model gets trained on for each optimization iteration. Should be a power of 2 (64, 128, 256 fx) for speed up.
    :param epochs: Specifies how many times the model learns from all the training data.
    :param learning_rate: a parameter that controls how much the weights and biases changes with each step. In other words: How fast the model learns.
    :param error_function: determines which function to use to calculate the error between predicted and actual values.
    :return:

    """

    print("\n")

    inputs, labels_npy, mapping = load_data(mapping_json, labels_npy, features_npy)

    if num_classes is not None:
        labels_npy = labels_npy[labels_npy < num_classes]
        inputs = inputs[0:len(labels_npy)]

    skf, x, x_test, y, y_test = cross_val(inputs=inputs, labels=labels_npy, test_size=0.20, folds=folds)

    if sub_set is not None:
        x = x[:sub_set]
        y = y[:sub_set]
        x_test = x_test[:int(sub_set * 0.20)]
        y_test = y_test[:int(sub_set * 0.20)]

    if prepocess:
        x = preprocess(x, model_name)
        x_test = preprocess(x_test, model_name)

    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_batches = test_dataset.batch(batch_size)

    input_shape = x[0].shape
    output_shape = len(mapping)

    scores = []
    max_fold_score = 0
    best_model = None
    best_fold = 0

    for i, (train, val) in enumerate(skf.split(x, y)):
        if type(specific_fold) == int and i != abs(specific_fold):
            continue
        else:
            print(f"Running fold {i}")
            x_train = x[train]
            x_val = x[val]
            y_train = y[train]
            y_val = y[val]
            train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
            val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
            train_batches = train_dataset.batch(batch_size)
            val_batches = val_dataset.batch(batch_size)

            fold_score, model = run_basic(train_batches, val_batches, input_shape, output_shape,
                                          model_name, learning_rate, error_function, epochs, batch_size,
                                          early_stop)

            if fold_score > max_fold_score:
                max_fold_score = fold_score
                best_model = model
                best_fold = i
            print(f"Fold {i} score = {fold_score}")
            scores.append(fold_score)

    mean_score = np.array(scores).mean()
    print(f"Mean fold score = {mean_score}")

    return mapping, mean_score, best_fold, best_model, test_dataset, max_fold_score, x, y


def run_final(x, y, mapping, model_name, batch_size, learning_rate, error_function, epochs, OUTPUT_DIR, preprocess_, test_ds=None):
    results_dir = os.path.join(OUTPUT_DIR, str(len(os.listdir(OUTPUT_DIR)) + 1))
    os.mkdir(results_dir)

    settings = {"batch_size": batch_size, "learning_rate": learning_rate, "error_func": error_function,
                "epochs": epochs}
    x, y, mapping = load_data(mapping, y, x)
    if preprocess_:
        x = preprocess(x, model_name)

    if test_ds is None:
        _, x_train, x_test, y_train, y_test = cross_val(x, y, test_size=0.20, folds=5, rgb=True)
        if preprocess_:
            x_test = preprocess(x_test, model_name)
        train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    else:
        train_ds = tf.data.Dataset.from_tensor_slices((x, y))
        test_ds = tf.data.experimental.load(test_ds)

    train_batches = train_ds.batch(batch_size)
    test_batches = test_ds.batch(batch_size)

    input_shape = np.array(list(train_batches.take(1).as_numpy_iterator())[0][0][0]).shape
    print(input_shape)
    output_shape = len(mapping)

    score, model = run_basic(train_batches=train_batches, val_batches=test_batches,
                             input_shape=input_shape, output_shape=output_shape,
                             model_name=model_name,
                             learning_rate=learning_rate, error_function=error_function,
                             epochs=epochs,
                             batch_size=batch_size, early_stop=True)



    model_dir = os.path.join(results_dir, model_name)
    os.mkdir(model_dir)
    print(f"Saving to {model_dir}")
    model.save(os.path.join(model_dir, "FINAL"))
    with open(os.path.join(model_dir, 'mapping.json'), 'w') as f:
        json.dump(mapping, f)
    with open(os.path.join(model_dir, 'settings.json'), 'w') as f:
        json.dump(settings, f)
    tf.data.experimental.save(test_ds, os.path.join(model_dir, 'test_dataset'))


def run_basic(train_batches, val_batches,
              input_shape, output_shape,
              model_name, learning_rate,
              error_function, epochs,
              batch_size, early_stop):

    build_model = getattr(architectures, model_name)
    model = build_model(input_shape, output_shape)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=error_function, metrics=["accuracy"])
    monitor = keras.callbacks.EarlyStopping(monitor="val_accuracy", min_delta=0.001, patience=30, verbose=1,
                                            restore_best_weights=True, mode="max")

    if early_stop:
        model.fit(x=train_batches, validation_data=val_batches, epochs=epochs, batch_size=batch_size,
                  callbacks=monitor, verbose=1)
    else:
        model.fit(x=train_batches, validation_data=val_batches, epochs=epochs, batch_size=batch_size, verbose=1)

    test_y = tf.concat(list(val_batches.map(lambda values, labels: labels)), axis=0)
    test_x = tf.concat(list(val_batches.map(lambda values, labels: values)), axis=0)
    y_score = model.predict(test_x)
    y_pred = np.argmax(y_score, axis=1)
    f1 = get_f1(test_y, y_pred)
    score = f1

    return score, model


def get_confusion_matrix(model, test_batches):
    test_labels = tf.concat(list(test_batches.map(lambda values, labels: labels)), axis=0)
    test_values = tf.concat(list(test_batches.map(lambda values, labels: values)), axis=0)
    predictions = model.predict(test_values)

    ConfusionMatrixDisplay.from_predictions(y_true=test_labels, y_pred=np.argmax(predictions, axis=1))

def get_f1(test_y, y_pred):
    return f1_score(y_true=test_y, y_pred=y_pred, average="weighted")


def get_classification_report(model, test_batches, output_dict=True):
    test_labels = tf.concat(list(test_batches.map(lambda values, labels: labels)), axis=0)
    test_values = tf.concat(list(test_batches.map(lambda values, labels: values)), axis=0)
    predictions = model.predict(test_values)

    return classification_report(test_labels, np.argmax(predictions, axis=1), output_dict=output_dict)
