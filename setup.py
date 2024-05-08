from setuptools import setup, find_packages
from typing import List


# constants
HYPHEN_E_DOT="-e ."


# functions
def get_requirements(file_path: str)->List[str]:
    # codoe here
    """
    This function will return a list of requirements after preparing that list using the provided file_path.
    """
    # reading all the files of the provided file_path and preparing requirements list
    with open(file_path, "r") as file:
        required_libs=file.readlines()
        # removing \n from the values
        required_libs=[req.replace("\n","") for req in required_libs]

        # removing HYPHEN_E_DOT from the values which we've written in the reqiurements to trigger setup.py
        if HYPHEN_E_DOT in required_libs:
            required_libs.remove(HYPHEN_E_DOT)

    return required_libs


# specifying setup details
setup(
    name="ML1 Project",
    version="0.0.1",
    author="Shaukat",
    author_email="iamshaukatalikhan@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")

)