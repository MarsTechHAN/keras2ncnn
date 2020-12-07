import setuptools

setuptools.setup(
    name="keras2ncnn",
    version="0.0.7",
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
    python_requires='>=3.7',
    install_requires=[
        'h5py>=2.10.0'
    ]
)