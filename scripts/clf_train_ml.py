import os
import sys
import json
import sklearn

from typing import Any, Dict

repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKOUT_PATH = repo_path
DATASET_PATH = os.path.join(repo_path, "datasets")

os.chdir(CHECKOUT_PATH)
sys.path.insert(0, CHECKOUT_PATH)


NAME_TO_MODEL_CLS: Dict[str, Any] = {
    "randomForest": sklearn.ensemble.RandomForestClassifier,
    "kNN": sklearn.neighbors.KNeighborsClassifier,
}

def run():
    import argparse

    parser = argparse.ArgumentParser(
        description="Test sklearn models on tasks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model",
        type=str,
        default="randomForest",
        choices=["randomForest", "kNN"],
        help="The model to use.",
    )

    parser.add_argument(
        "--model-params",
        type=lambda s: json.loads(s),
        default={},
        help=(
            "JSON dictionary containing model hyperparameters, if not using grid search these will"
            " be used."
        ),
    )
    parser.add_argument("--debug", dest="debug", action="store_true", help="Enable debug routines")
    args = parser.parse_args()

    run_and_debug(lambda: run_from_args(args), args.debug)


if __name__ == "__main__":
    run()
