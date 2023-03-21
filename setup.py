# This is so we can import from the parent directory when using notebooks
# Simply run pip install -e . from the root directory
import setuptools

setuptools.setup(
    name="my_package",
    version="0.0.1",
    description="A small example package",
    packages=setuptools.find_packages(),
    python_requires='>=3.7',
)