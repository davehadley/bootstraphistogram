from setuptools import setup

with open("README.md", "r") as readme:
    long_description = readme.read()

setup(name="bootstraphistogram",
      version="0.2",
      description="Poisson bootstrap histogram.",
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="https://github.com/davehadley/bootstraphistogram",
      author="David Hadley",
      author_email="d.r.hadley@warwick.ac.uk",
      license="MIT",
      packages=["bootstraphistogram"],
      install_requires=["boost-histogram>=0.7.0", "numpy>=1.18.5", "matplotlib>=3.2.0"],
      zip_safe=True,
      classifiers=[
          "Programming Language :: Python :: 3 :: Only",
          "License :: OSI Approved :: MIT License",
          "Development Status :: 2 - Pre-Alpha",
          "Operating System :: POSIX",
          "Intended Audience :: Science/Research",
      ],
      python_requires=">=3.6",
      )
