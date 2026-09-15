import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def calculate_metrics(
    y_true,
    y_pred
):
    """
    Calculate classification metrics.
    """

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def get_confusion_matrix(
    y_true,
    y_pred
):
    """
    Return the confusion matrix.
    """

    return confusion_matrix(
        y_true,
        y_pred
    )


def print_classification_report(
    y_true,
    y_pred
):
    """
    Print a detailed classification report.
    """

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=[
                "Negative",
                "Positive"
            ],
            zero_division=0
        )
    )