import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bisturi",
    version="0.0.1",
    author="Riccardo Massidda",
    author_email="riccardo.massidda@phd.unipi.it",
    description="Framework for neural models inspection.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rmassidda/bisturi/",
    project_urls={
        "Bug Tracker": "https://github.com/rmassidda/bisturi/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
    install_requires=[
        'imageio',
        'numpy',
        'Pillow',
        'scipy',
        'torch',
        'torchinfo',
        'torchvision',
        'sphinx',
        'sphinxcontrib-napoleon',
        'sphinx_rtd_theme',
        'nltk',
        'tqdm'
        ]
)
