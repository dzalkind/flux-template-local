import os
from typing import List

from .pipeline import Pipeline


def run_pipeline(mode: str = 'prod', input_files: List[str] = []):
    """-------------------------------------------------------------------
    Run the pipeline on the given input files.  If no input files are
    provided, then run the pipeline on any files in the data/inputs
    directory (this is used for testing).
    -------------------------------------------------------------------"""

    # Load the config files
    package_dir = os.path.dirname(os.path.realpath(__file__))
    project_dir = os.path.dirname(package_dir)
    data_dir = os.path.join(project_dir, 'data')
    config_dir = os.path.join(project_dir, 'config')

    pipeline_config = os.path.join(config_dir, 'pipeline_config.yml')
    storage_config_filename = f"storage_config_{mode}.yml"
    storage_config = os.path.join(config_dir, storage_config_filename)

    # Create pipeline
    pipeline = Pipeline(pipeline_config, storage_config)

    # If no files specified, then use the data/inputs directory
    if len(input_files) == 0:
        input_files = [os.path.join(data_dir, 'inputs')]

    # Run the pipeline on the given files.
    for file_path in input_files:
        if os.path.isdir(file_path):
            for root, dirs, files in os.walk(file_path):
                for name in files:
                    path = os.path.join(root, name)
                    pipeline.run(path)
        else:
            pipeline.run(file_path)
