from typing import Optional

import numpy as np
from tsdat.qc import QualityChecker, QualityHandler


class DummyQCTest(QualityChecker):
    """-------------------------------------------------------------------
    Class containing placeholder code to perform a single QC test on a
    Dataset variable.

    See https://tsdat.readthedocs.io/ for more QC examples.
    -------------------------------------------------------------------"""

    def run(self, variable_name: str) -> Optional[np.ndarray]:
        """-------------------------------------------------------------------
        Test a dataset's variable to see if it passes a quality check.
        These tests can be performed on the entire variable at one time by
        using xarray vectorized numerical operators.

        Args:
            variable_name (str):  The name of the variable to check

        Returns:
            np.ndarray | None: If the test was performed, return a
            ndarray of the same shape as the variable. Each value in the
            data array will be either True or False, depending upon the
            results of the test.  True means the test failed.  False means
            it succeeded.

            Note that we are using an np.ndarray instead of an xr.DataArray
            because the DataArray contains coordinate indexes which can
            sometimes get out of sync when performing np arithmectic vector
            operations.  So it's easier to just use numpy arrays.

            If the test was skipped for some reason (i.e., it was not
            relevant given the current attributes defined for this dataset),
            then the run method should return None.
        -------------------------------------------------------------------"""

        # Just return an array of all False of same shape as the variable
        results_array = np.full_like(self.ds[variable_name].data, False, dtype=bool)

        return results_array


class DummyErrorHandler(QualityHandler):
    """-------------------------------------------------------------------
    Class containing placeholder code for a custom error handler.

    See https://tsdat.readthedocs.io/ for more QC examples.
    -------------------------------------------------------------------"""

    def run(self, variable_name: str, results_array: np.ndarray):
        """-------------------------------------------------------------------
        Perform a follow-on action if a qc test fails.  This can be used to
        correct data if needed (such as replacing a bad value with missing value,
        emailing a contact persion, adding additional metadata, or raising an
        exception if the failure constitutes a critical error).

        Args:
            variable_name (str): Name of the variable that failed
            results_array (np.ndarray)  : An array of True/False values for
            each data value of the variable.  True means the test failed.
        -------------------------------------------------------------------"""
        print(f"QC test failed for variable {variable_name}")
