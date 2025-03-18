from typing import Tuple, Sequence
from data.dataset import Dataset
import numpy as np
import pandas as pd


def read_csv(filename, sep=',',features=True, label=True):
    """
        Arguments:
        filename : Name/path of the file.
        sep : Value separator. Default is ','.
        features: Does the file include feature names? Default is True.
        label : bool : Does the file include a label (y)? Default is True.

        Returns:
         The dataset object.
        """
    df = pd.read_csv(filename, sep=sep, header=0 if features else None)

    if label:
        X = df.iloc[:, :-1]  # All columns except the last one
        y = df.iloc[:, -1]  # Last column
    else:
        X = df
        y = None

    return Dataset(X, y)

def write_csv(filename,dataset, sep=',',features=True, label=True):
    """
    Arguments:
    filename: Name/path of the file.
    dataset: The Dataset object to write to a CSV.
    sep: Value separator.Default is ','.
    features: Boolean. Should feature names be included? Default is True.
    label: Boolean. Should a label (y) be included? Default is True.
    Expected Output:
    Writes the file with the specified arguments."""

    df = dataset.features.copy()

    if label and  dataset.label is not None:

        df["label"]= dataset.label

    df.to_csv(filename,sep=sep, index=False, header=features)

    print(f"File written successfully to {filename}")
