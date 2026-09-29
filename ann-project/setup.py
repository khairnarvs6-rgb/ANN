from setuptools import setup, find_packages

setup(
    name="ann-project",
    version="1.0.0",
    description="Artificial Neural Network from scratch with NumPy",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=["numpy", "pandas", "scikit-learn", "matplotlib", "pyyaml"],
)
