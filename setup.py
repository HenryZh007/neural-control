from setuptools import setup, Extension
import glob
import os

# Use glob to find the actual .so file with cp310-... in its name
so_file = glob.glob(os.path.join('nn_der', 'nn_der*.so'))[0]  # Adjust path as needed

setup(
    name='nn_der',
    version='0.1',
    packages=['nn_der'],
    package_dir={'nn_der': 'nn_der'},
    package_data={'nn_der': [os.path.basename(so_file)]},  # Include the .so file
)