import os
from typing import Dict

import cmocean
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from tsdat.pipeline import IngestPipeline
from tsdat.utils import DSUtil

example_dir = os.path.abspath(os.path.dirname(__file__))
style_file = os.path.join(example_dir, "styling.mplstyle")
plt.style.use(style_file)


class Pipeline(IngestPipeline):
    """Example tsdat ingest pipeline used to process lidar instrument data from
    a buoy stationed at Morro Bay, California.

    See https://tsdat.readthedocs.io/ for more on configuring tsdat pipelines.
    """

    def hook_customize_raw_datasets(self, raw_dataset_mapping: Dict[str, xr.Dataset]) -> Dict[str, xr.Dataset]:
        """-------------------------------------------------------------------
        Hook to allow for user customizations to one or more raw xarray Datasets
        before they merged and used to create the standardized dataset.  The
        raw_dataset_mapping will contain one entry for each file being used
        as input to the pipeline.  The keys are the standardized raw file name,
        and the values are the datasets.

        This method would typically only be used if the user is combining
        multiple files into a single dataset.  In this case, this method may
        be used to correct coordinates if they don't match for all the files,
        or to change variable (column) names if two files have the same
        name for a variable, but they are two distinct variables.

        This method can also be used to check for unique conditions in the raw
        data that should cause a pipeline failure if they are not met.

        This method is called before the inputs are merged and converted to
        standard format as specified by the config file.

        Args:
        ---
            raw_dataset_mapping (Dict[str, xr.Dataset])     The raw datasets to
                                                            customize.

        Returns:
        ---
            Dict[str, xr.Dataset]: The customized raw dataset.
        -------------------------------------------------------------------"""
        return raw_dataset_mapping

    def hook_customize_dataset(self, dataset: xr.Dataset, raw_mapping: Dict[str, xr.Dataset]) -> xr.Dataset:
        """-------------------------------------------------------------------
        Hook to allow for user customizations to the standardized dataset such
        as inserting a derived variable based on other variables in the
        dataset.  This method is called immediately after the apply_corrections
        hook and before any QC tests are applied.

        Args:
        ---
            dataset (xr.Dataset): The dataset to customize.
            raw_mapping (Dict[str, xr.Dataset]):    The raw dataset mapping.

        Returns:
        ---
            xr.Dataset: The customized dataset.
        -------------------------------------------------------------------"""
        
        # Compress row of variables in input into variables dimensioned by time and height
        for raw_filename, raw_dataset in raw_mapping.items():
            if "nwtc.flux_z01" in raw_filename:
                raw_categories = ["U_ax","V_ax","W_ax","Ts"]
                output_var_names = ["U_ax", "V_ax", "W_ax", "Ts"]
                heights = dataset.height.data
                for category, output_name in zip(raw_categories, output_var_names):
                    var_names = [f"{category}_{height}m" for height in heights]
                    var_data = [raw_dataset[name].data for name in var_names]
                    var_data = np.array(var_data).transpose()
                    dataset[output_name].data = var_data

                # wind speed and direction, relative to U direction
                dataset['wind_speed'] = np.sqrt(dataset.U_ax**2 + dataset.V_ax**2)
                direction_raw = np.degrees(np.arctan2(dataset.V_ax,dataset.U_ax))
                dataset['wind_direction'] = (dataset.dims,np.where(direction_raw<0,direction_raw+360,direction_raw).T)

        return dataset

    def hook_finalize_dataset(self, dataset: xr.Dataset) -> xr.Dataset:
        """-------------------------------------------------------------------
        Hook to apply any final customizations to the dataset before it is
        saved. This hook is called after quality tests have been applied.

        Args:
            dataset (xr.Dataset): The dataset to finalize.

        Returns:
            xr.Dataset: The finalized dataset to save.
        -------------------------------------------------------------------"""
        return dataset

    def hook_generate_and_persist_plots(self, dataset: xr.Dataset) -> None:
        """-------------------------------------------------------------------
        Hook to allow users to create plots from the xarray dataset after
        processing and QC have been applied and just before the dataset is
        saved to disk.

        To save on filesystem space (which is limited when running on the
        cloud via a lambda function), this method should only
        write one plot to local storage at a time. An example of how this
        could be done is below:

        ```
        filename = DSUtil.get_plot_filename(dataset, "sea_level", "png")
        with self.storage._tmp.get_temp_filepath(filename) as tmp_path:
            fig, ax = plt.subplots(figsize=(10,5))
            ax.plot(dataset["time"].data, dataset["sea_level"].data)
            fig.save(tmp_path)
            storage.save(tmp_path)

        filename = DSUtil.get_plot_filename(dataset, "qc_sea_level", "png")
        with self.storage._tmp.get_temp_filepath(filename) as tmp_path:
            fig, ax = plt.subplots(figsize=(10,5))
            DSUtil.plot_qc(dataset, "sea_level", tmp_path)
            storage.save(tmp_path)
        ```

        Args:
        ---
            dataset (xr.Dataset):   The xarray dataset with customizations and
                                    QC applied.
        -------------------------------------------------------------------"""

        def format_time_xticks(ax, start=4, stop=21, step=4, date_format="%H-%M"):
            ax.xaxis.set_major_locator(mpl.dates.HourLocator(byhour=range(start, stop, step)))
            ax.xaxis.set_major_formatter(mpl.dates.DateFormatter(date_format))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center')

        def add_colorbar(ax, plot, label):
            cb = plt.colorbar(plot, ax=ax, pad=0.01)
            cb.ax.set_ylabel(label, fontsize=12)
            cb.outline.set_linewidth(1)
            cb.ax.tick_params(size=0)
            cb.ax.minorticks_off()
            return cb

        ds = dataset
        date = pd.to_datetime(ds.time.data[0]).strftime('%d-%b-%Y')

        # Colormaps to use
        wind_cmap = cmocean.cm.deep_r
        avail_cmap = cmocean.cm.amp_r

        # Create the first plot - Lidar Wind Speeds at several elevations
        filename = DSUtil.get_plot_filename(dataset, "wind_speed_and_dir", "png")
        with self.storage._tmp.get_temp_filepath(filename) as tmp_path:

            # Create the figure and axes objects
            fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(14, 8), constrained_layout=True)
            fig.suptitle(f"Wind Speed and Direction Time Series at {ds.attrs['location_meaning']} on {date}")

            # Select heights to plot
            heights = [3,7]

            # Plot the wind speed
            for i, height in enumerate(heights):
                velocity = ds.wind_speed.sel(height=height)
                velocity.plot(ax=ax[0], linewidth=2, c=wind_cmap(i / len(heights)), label=f"{height} m")

            # Set the labels and ticks
            format_time_xticks(ax[0])
            ax[0].legend(facecolor="white", ncol=len(heights), bbox_to_anchor=(1, -0.05))
            ax[0].set_title("")  # Remove bogus title created by xarray
            ax[0].set_xlabel("Time (UTC)")
            ax[0].set_ylabel(r"Wind Speed (ms$^{-1}$)")

            # Plot the wind direction
            for i, height in enumerate(heights):
                direction = ds.wind_direction.sel(height=height)
                direction.plot(ax=ax[1], linewidth=2, c=wind_cmap(i / len(heights)), label=f"{height} m")

            # Set the labels and ticks
            format_time_xticks(ax[1])
            ax[1].legend(facecolor="white", ncol=len(heights), bbox_to_anchor=(1, -0.05))
            ax[1].set_title("")  # Remove bogus title created by xarray
            ax[1].set_xlabel("Time (UTC)")
            ax[1].set_ylabel(r"Wind Direction (deg. relative to U)")

            # Save the figure
            fig.savefig(tmp_path, dpi=100)
            self.storage.save(tmp_path)
            plt.close()

        return
