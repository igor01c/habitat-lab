from setuptools import setup, find_packages

setup(
    name="habitat_object_benchmark",
    version="0.1.0",
    packages=["habitat_object_benchmark"]
    + [f"habitat_object_benchmark.{p}" for p in ["checks", "utils", "scripts"]],
    install_requires=["pandas", "pyarrow", "tqdm", "numpy"],
)
