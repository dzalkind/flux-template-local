#!/usr/bin/env python
"""-----------------------------------------------------------------------
  Python script for running the custom tsdat pipeline defined by this
  project.
---------------------------------------------------------------------------"""
import argparse
import sys
import os

# Add the project directory to the pythonpath
project_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, project_dir)

from pipeline.runner import run_pipeline


def main():
    """-------------------------------------------------------------------
    Main function.
    -------------------------------------------------------------------"""

    # Parse arguments - a file or list of files
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", default='dev', help="Identify the configuration to use.  Default is dev.")
    parser.add_argument('file', nargs='*', type=str)
    args = parser.parse_args()
    files = []

    for f in args.file:
        files.append(f)

    # run the pipeline
    run_pipeline(mode=args.mode, input_files=files)


if __name__ == "__main__":
    main()
