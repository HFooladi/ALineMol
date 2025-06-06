"""
Script for training binary classification models using scikit-learn and saving the results.
"""

# import required libraries
import os  # for file operations
import sys  # for system operations
import json  # for json operations
import numpy as np  # for numerical operations
import pandas as pd  # for data operations
import datamol as dm  # for molecular operations
import joblib  # for saving models
from typing import Any, Dict  # for type hinting

from sklearn import metrics  # for metrics
from sklearn.ensemble import RandomForestClassifier  # for random forest classifier
from sklearn.neural_network import MLPClassifier  # for multi-layer perceptron classifier
from sklearn.neighbors import KNeighborsClassifier  # for k-nearest neighbors classifier
from sklearn.svm import SVC  # for support vector classifier
from xgboost import XGBClassifier  # for xgboost classifier
from lightgbm import LGBMClassifier  # for lightgbm classifier

# import required functions from the repository
from alinemol.utils import get_configure, init_trial_path
from alinemol.utils.split_utils import sklearn_random_split, sklearn_stratified_random_split
from alinemol.utils.logger_utils import logger
from alinemol.utils.metric_utils import compute_binary_task_metrics

# set the path to the repository
repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKOUT_PATH = repo_path
DATASET_PATH = os.path.join(repo_path, "datasets")

# change the directory to the repository path
os.chdir(CHECKOUT_PATH)
sys.path.insert(0, CHECKOUT_PATH)


# dictionary to map model names to model classes
NAME_TO_MODEL_CLS: Dict[str, Any] = {
    "randomForest": RandomForestClassifier,
    "kNN": KNeighborsClassifier,
    "SVM": SVC,
    "MLP": MLPClassifier,
    "XGB": XGBClassifier,
    "LightGBM": LGBMClassifier,
}


# main function to train the model
def main(args: Dict, exp_config: Dict):
    # get data in to form for sklearn
    assert args["split_ratio"] is not None
    assert args["model"] is not None
    # read the csv file
    df = pd.read_csv(args["csv_path"])
    assert args["smiles_column"] in df.columns
    assert "label" in df.columns

    # get the SMILES and labels
    smiles = df[args["smiles_column"]]
    labels = np.array(df["label"])

    # convert SMILES to fingerprints
    fps = np.array([dm.to_fp(dm.to_mol(smi)) for smi in smiles])

    # split the data into train, validation and test sets
    train_ratio, val_ratio, test_ratio = map(float, args["split_ratio"].split(","))
    assert train_ratio + val_ratio + test_ratio == 1
    split_ratio = (train_ratio, val_ratio, test_ratio)
    if args["split"] == "stratified_random":
        X_train, X_val, X_test, y_train, y_val, y_test = sklearn_stratified_random_split(fps, labels, split_ratio)
    elif args["split"] == "random":
        X_train, X_val, X_test, y_train, y_val, y_test = sklearn_random_split(fps, labels, split_ratio)

    # Set up directory for saving results
    args = init_trial_path(args)

    # Initialize the model with the hyperparameters
    model_cls = NAME_TO_MODEL_CLS[args["model"]]
    if bool(args["model_params"]):
        logger.info("model_params: ", args["model_params"])
        model = model_cls(**args["model_params"])
    else:
        logger.info(f"exp_config: {exp_config}")
        model = model_cls(**exp_config)

    logger.info("Model architecture: {}".format(args["model"]))
    logger.info(model)
    # Train the model
    model.fit(X_train, y_train)

    print("Model trained")
    print("Training Accuracy : ", metrics.accuracy_score(y_train, model.predict(X_train)) * 100)
    print("Validation Accuracy : ", metrics.accuracy_score(y_val, model.predict(X_val)) * 100)

    # Compute test results:
    y_predicted_true_probs_val = model.predict_proba(X_val)[:, 1]
    y_predicted_true_probs_test = model.predict_proba(X_test)[:, 1]
    val_metrics = compute_binary_task_metrics(y_predicted_true_probs_val, y_val)
    test_metrics = compute_binary_task_metrics(y_predicted_true_probs_test, y_test)

    # save the results as a text file
    with open(args["trial_path"] + "/eval.txt", "w") as f:
        f.write("Best val {}: {}\n".format(args["metric"], val_metrics.roc_auc))
        f.write("Test {}: {}\n".format("accuracy_score", test_metrics.acc))
        f.write("Test {}: {}\n".format("roc_auc_score", test_metrics.roc_auc))
        f.write("Test {}: {}\n".format("pr_auc_score", test_metrics.avg_precision))

    # save the metrics as a csv file
    reports = {
        f"{args['metric']}_val": val_metrics.roc_auc,
        "acc": test_metrics.acc,
        "roc_auc": test_metrics.roc_auc,
        "pr_auc": test_metrics.avg_precision,
        "model": args["model"],
        "test_size": len(y_test),
        "val_size": len(y_val),
        "train_size": len(y_train),
    }
    df = pd.DataFrame(reports, index=[0])
    df.to_csv(args["trial_path"] + "/metrics.csv")

    # adding filepaths to the experiment config file and saving it
    exp_config.update({"filepath": args["csv_path"]})
    logger.info("experimetns_config: {}".format(exp_config))
    with open(args["trial_path"] + "/configure.json", "w") as f:
        json.dump(exp_config, f, indent=2)

    # save the model
    joblib.dump(model, args["trial_path"] + "/model.joblib")


if __name__ == "__main__":
    from argparse import ArgumentParser  # for argument parsing

    parser = ArgumentParser("Mchine Learning Binary Classification with scikit-learn")
    parser.add_argument(
        "-mo",
        "--model",
        type=str,
        default="randomForest",
        choices=["randomForest", "kNN", "SVM", "XGB", "LightGBM", "MLP"],
        help="The model to use.",
    )
    parser.add_argument(
        "--model_params",
        type=lambda s: json.loads(s),
        default={},
        help="JSON dictionary containing model hyperparameters.",
    )
    parser.add_argument("-c", "--csv_path", type=str, required=True, help="Path to a csv file for loading a dataset")
    parser.add_argument(
        "-sc", "--smiles_column", type=str, required=True, help="Header for the SMILES column in the CSV file"
    )
    parser.add_argument(
        "-s",
        "--split",
        choices=["scaffold_decompose", "scaffold_smiles", "random", "stratified_random"],
        default="stratified_random",
        help="Dataset splitting method (default: stratified_random). For scaffold "
        "split based on rdkit.Chem.AllChem.MurckoDecompose, "
        "use scaffold_decompose. For scaffold split based on "
        "rdkit.Chem.Scaffolds.MurckoScaffold.MurckoScaffoldSmiles, "
        "use scaffold_smiles.",
    )
    parser.add_argument(
        "-sr",
        "--split_ratio",
        default="0.72,0.08,0.2",
        type=str,
        help="Proportion of the dataset to use for training, validation and test, (default: 0.7,0.1,0.2)",
    )
    parser.add_argument(
        "-me",
        "--metric",
        choices=["acc", "roc_auc", "pr_auc"],
        default="roc_auc",
        help="Metric for evaluation (default: roc_auc)",
    )
    parser.add_argument(
        "-p",
        "--result_path",
        type=str,
        default="classification_results",
        help="Path to save training results (default: classification_results)",
    )
    args: Dict = vars(parser.parse_args())
    exp_config: Dict = get_configure(args["model"])  # get the configuration (hyperparameters) for the model
    main(args, exp_config)  # train the model
