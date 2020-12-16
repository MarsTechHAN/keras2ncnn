import setuptools

setuptools.setup(
    name="keras2ncnn",
    version="0.1.2",
    author="Han Xiao",
    author_email="hansh-sz@hotmail.com",
    description="A keras h5df to ncnn converter",
    url="https://github.com/MarsTechHAN/keras2ncnn",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
    install_requires=[
        'h5py>=2.10.0',
        'virtualenv>=15.0.0'
    ]
)
