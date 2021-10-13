import xarray as xr
import pandas as pd
from tsdat.io import AbstractFileHandler
from tsdat import Config


class DummyFileHandler(AbstractFileHandler):
    """-------------------------------------------------------------------
    Custom file handler for reading *.sta files.

    See https://tsdat.readthedocs.io/ for more file handler examples.
    -------------------------------------------------------------------"""

    def write(self, ds: xr.Dataset, filename: str, config: Config, **kwargs):
        """-------------------------------------------------------------------
        Classes derived from the FileHandler class can implement this method
        to save to a custom file format.

        Args:
            ds (xr.Dataset): The dataset to save.
            filename (str): An absolute or relative path to the file including
                            filename.
            config (Config, optional):  Optional Config object. Defaults to
                                        None.
        -------------------------------------------------------------------"""
        raise NotImplementedError("Error: this file format should not be used to write to.")

    def read(self, filename: str, **kwargs) -> xr.Dataset:
        """-------------------------------------------------------------------
        Classes derived from the FileHandler class can implement this method.
        to read a custom file format into a xr.Dataset object.

        Args:
            filename (str): The path to the file to read in.

        Returns:
            xr.Dataset: An xr.Dataset object
        -------------------------------------------------------------------"""

        df = pd.read_csv(filename, sep="\t", header=41, index_col=False)
        return df.to_xarray()
