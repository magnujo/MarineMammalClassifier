from MLModels import architectures as arc
from Preprocessing.to_npy import save_specs
import os


audio_root_dir_path="data/audio",
specs_dir = "data/processed"

save_specs(
    audio_root_dir_path=audio_root_dir_path,
    destination_dir=specs_dir,
    duration_per_track=1.0
)

from MLModels.CNN_tensor import run, get_confusion_matrix, get_classification_report

mapping_path  = os.path.join(specs_dir, "mapping.json")
labels_path   = os.path.join(specs_dir, "labels.npy")
features_path = os.path.join(specs_dir, "spectrograms.npy")

mapping, mean_f1, best_fold, best_model, test_dataset, best_f1, x_train_part, y_train_part = run(
    mapping_json=mapping_path,
    labels_npy=labels_path,
    features_npy=features_path,
    batch_size=64,
    epochs=100,
    learning_rate=1e-4,
    model_name="MyCNN",  # must exist as architectures.MyCNN
    error_function="sparse_categorical_crossentropy",
    early_stop=True,
    folds=5,
    specific_fold=None,     # set e.g. 0 to run only fold 0
    num_classes=None,       # set e.g. 10 to keep labels < 10
    prepocess=False,        # keep False unless using keras.applications model_name
    sub_set=None            # set e.g. 2000 for quick debugging
)

print("Mean CV weighted F1:", mean_f1)
print("Best fold:", best_fold, "Best fold F1:", best_f1)

# Evaluate the returned best_model on the held-out test set
test_batches = test_dataset.batch(64)

report = get_classification_report(best_model, test_batches, output_dict=False)
print(report)

get_confusion_matrix(best_model, test_batches)  # shows a plot








