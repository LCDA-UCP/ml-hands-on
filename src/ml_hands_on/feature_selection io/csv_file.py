from typing import Tuple, Sequence

import numpy as np
import pandas as pd

"""Arguments:
filename: Name/path of the file.
sep: Value separator.
features: Boolean. Does the file include feature names?
label: Boolean. Does the file include a label (y)? (Assume the label is the last column if present.)
Expected Output:
Returns a Dataset object."""

def read_csv(filename, sep=',',features=True, label=True):
    """
        Arguments:
        filename : str : Name/path of the file.
        sep : str : Value separator. Default is ','.
        features : bool : Does the file include feature names? Default is True.
        label : bool : Does the file include a label (y)? Default is True.

        Returns:
        dataset : pandas DataFrame : The dataset object read from the CSV file.
        """