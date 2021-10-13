import os
import sys
import shutil
import unittest

# Add the project directory to the pythonpath
test_dir = os.path.dirname(os.path.realpath(__file__))
project_dir = os.path.dirname(test_dir)
sys.path.insert(0, project_dir)

from pipeline.runner import run_pipeline


def _delete_dir(folder_path):
    if os.path.isdir(folder_path):
        shutil.rmtree(folder_path)


class TestPipeline(unittest.TestCase):

    def test_pipeline(self):
        # Run the pipeline via the runner
        run_pipeline(mode='dev')


if __name__ == '__main__':
    unittest.main()
