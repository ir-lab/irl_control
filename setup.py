from setuptools import setup, find_packages
import runpy
import pathlib

root = pathlib.Path(__file__).parent
version = runpy.run_path(str(root / "irl_control" / "version.py"))["version"]

setup(
    name='irl_control',
    version=version,
    description='Control Suite for Bi-Manual Manipulation tasks in Mujoco',
    url='https://github.com/ir-lab/irl_control',
    author='Michael Drolet, Ravi Swaroop',
    author_email='mdrolet@asu.edu',
    license='MIT',
    packages=find_packages()
)