import pandas as pd

from ..models.parse import Format


def load_dataframe(bytes, format) -> pd.DataFrame:
    if format == Format.PARQUET:
        data = pd.read_parquet(bytes)
    elif format == Format.CSV:
        data = pd.read_csv(bytes)
    elif format == Format.PICKLE:
        data = pd.read_pickle(bytes)
    elif format == Format.JSON:
        data = pd.read_json(bytes)
    else:
        raise ValueError("Unrecognized data_format {}!".format(format))
    return data
