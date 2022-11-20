import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="wideboost",
    version="0.4.0",
    author="Michael Horrell",
    author_email="mthorrell@github.com",
    description="Implements Wide Boosting functions for popular boosting packages",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mthorrell/wideboost",
    packages=setuptools.find_packages(),
    install_requires=['numpy', 'xgboost'],
    extras_require={
        'scikit-learn': ['scikit-learn'],
        'lightgbm': ['lightgbm'],
        'shap': ['shap']
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
