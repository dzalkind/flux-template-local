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
            if ".sta" in raw_filename:
                raw_categories = ["Wind Speed (m/s)", "Wind Direction (�)", "Data Availability (%)"]
                output_var_names = ["wind_speed", "wind_direction", "data_availability"]
                heights = dataset.height.data
                for category, output_name in zip(raw_categories, output_var_names):
                    var_names = [f"{height}m {category}" for height in heights]
                    var_data = [raw_dataset[name].data for name in var_names]
                    var_data = np.array(var_data).transpose()
                    dataset[output_name].data = var_data

        # Apply correction to buoy at morro bay -- wind direction is off by 180 degrees
        if "morro" in dataset.attrs["datastream_name"]:
            new_direction = dataset["wind_direction"].data + 180
            new_direction[new_direction >= 360] -= 360
            dataset["wind_direction"].data = new_direction
            dataset["wind_direction"].attrs["corrections_applied"] = "Applied +180 degree calibration factor."

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
        filename = DSUtil.get_plot_filename(dataset, "wind_speeds", "png")
        with self.storage._tmp.get_temp_filepath(filename) as tmp_path:

            # Create the figure and axes objects
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14, 8), constrained_layout=True)
            fig.suptitle(f"Wind Speed Time Series at {ds.attrs['location_meaning']} on {date}")

            # Select heights to plot
            heights = [40, 90, 140, 200]

            # Plot the data
            for i, height in enumerate(heights):
                velocity = ds.wind_speed.sel(height=height)
                velocity.plot(ax=ax, linewidth=2, c=wind_cmap(i / len(heights)), label=f"{height} m")

            # Set the labels and ticks
            format_time_xticks(ax)
            ax.legend(facecolor="white", ncol=len(heights), bbox_to_anchor=(1, -0.05))
            ax.set_title("")  # Remove bogus title created by xarray
            ax.set_xlabel("Time (UTC)")
            ax.set_ylabel(r"Wind Speed (ms$^{-1}$)")

            # Save the figure
            fig.savefig(tmp_path, dpi=100)
            self.storage.save(tmp_path)
            plt.close()

        filename = DSUtil.get_plot_filename(dataset, "wind_speed_and_direction", "png")
        with self.storage._tmp.get_temp_filepath(filename) as tmp_path:

            # Reduce dimensionality of dataset for plotting quivers
            ds_1H: xr.Dataset = ds.resample(time="1H").nearest()

            # Calculations for contour plots
            levels = 30

            # Calculations for quiver plot
            qv_slice = slice(1, None)  # Skip first to prevent weird overlap with axes borders
            qv_degrees = ds_1H.wind_direction.data[qv_slice].transpose()
            qv_theta = (qv_degrees + 90) * (np.pi / 180)
            X, Y = ds_1H.time.data[qv_slice], ds_1H.height.data
            U, V = np.cos(-qv_theta), np.sin(-qv_theta)

            # Create figure and axes objects
            fig, axs = plt.subplots(nrows=2, figsize=(14, 8), constrained_layout=True)
            fig.suptitle(f"Wind Speed and Direction at {ds.attrs['location_meaning']} on {date}")

            # Make top subplot -- contours and quiver plots for wind speed and direction
            csf = ds.wind_speed.plot.contourf(ax=axs[0], x="time", levels=levels, cmap=wind_cmap, add_colorbar=False)
            axs[0].quiver(X, Y, U, V, width=0.002, scale=60, color="white", pivot='middle', zorder=10)
            add_colorbar(axs[0], csf, r"Wind Speed (ms$^{-1}$)")

            # Make bottom subplot -- heatmap for data availability
            da = ds.data_availability.plot(ax=axs[1], x="time", cmap=avail_cmap, add_colorbar=False, vmin=0, vmax=100)
            add_colorbar(axs[1], da, "Availability (%)")

            # Set the labels and ticks
            for i in range(2):
                format_time_xticks(axs[i])
                axs[i].set_xlabel("Time (UTC)")
                axs[i].set_ylabel("Height ASL (m)")

            # Save the figure
            fig.savefig(tmp_path, dpi=100)
            self.storage.save(tmp_path)
            plt.close()

        return
