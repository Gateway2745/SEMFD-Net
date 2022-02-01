import setuptools

with open("readme.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="SEMFD-Net",
    version="0.0.1",
    description="Stacked Ensemble Model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Gateway2745/SEMFD-Net/",
    project_urls={
        "Bug Tracker": "https://github.com/Gateway2745/SEMFD-Net/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    packages=setuptools.find_packages(),
    install_requires = ["resnest==0.0.5", "omegaconf==2.1.1", "mlflow==1.19.0", 'pytorch-lightning==1.5.9', 'torch==1.10.2', 'torchmetrics==0.7.0','timm==0.4.12'],
    python_requires=">=3.7",
)