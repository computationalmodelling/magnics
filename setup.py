import setuptools

with open("README.rst") as f:
    readme = f.read()

setuptools.setup(
    name="magnics",
    version="0.1",
    description="Magnics is Python micromagnetics software using FEniCS.",
    long_description=readme,
    url="https://github.com/computationalmodelling/magnics",
    author="Leoni Breth, and Hans Fangohr",
    author_email="fangohr@soton.ac.uk",
    packages=setuptools.find_packages(),
    python_requires=">=3.5",
    classifiers=["Development Status :: 3 - Alpha",
                 "License :: OSI Approved :: BSD License",
                 "Topic :: Scientific/Engineering :: Physics",
                 "Intended Audience :: Science/Research",
                 "Programming Language :: Python :: 3 :: Only"]
)
