import pandas as pd

from ml_hands_on.data import Dataset


def read_data_file(filename, sep=',', label=True):
    """
        Arguments:
        filename : Name/path of the file.
        sep : Value separator. Default is ','
        label : bool : Does the file include a label (y)? Default is True.

        Returns:
         The dataset object.
        """
    file_extension = filename.split(".")[-1].lower()

    # CSV e TXT
    if file_extension in ["csv", "txt"]:
        df = pd.read_csv(filename, sep=sep)

    # JSON
    elif file_extension == "json":
        df = pd.read_json(filename)

    # Excel
    elif file_extension in ["xls", "xlsx"]:
        df = pd.read_excel(filename)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")


    if label and df.shape[1] > 1:
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
    else:
        X = df
        y = None

    return Dataset(X, y)


def read_data_file(filename,dataset, sep, label=True):
    """
       Writes a Dataset object to a file.

       Arguments:
       filename : str : Name/path of the file.
       dataset : Dataset : The Dataset object to write to a file.
       sep : str : Value separator.
       label : bool : Should a label (y) be included? Default is True.

       Writes the file with the specified format.
       """

    df = pd.DataFrame(dataset.X, columns= dataset.features)

    if label and  dataset.label is not None:

        df["label"]= dataset.label

    file_extension = filename.split('.')[-1].lower()

    if file_extension in ["csv", "txt"]:
        df.to_csv(filename, sep=sep, index=False)

    elif file_extension == 'json':
        df.to_json(filename, orient='records', lines=True)

    elif file_extension in ["xls", "xlsx"]:
        df.to_excel(filename, index=False)

    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

    print(f"File written successfully to {filename}")