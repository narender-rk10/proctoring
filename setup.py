from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="proctoring",
    version="0.0.1",
    author="Narender Keswani",
    author_email="narender.rk10@gmail.com",
    description="Proctoring library",
    keywords=['proctoring','ai-proctoring','gaze-tracking'],
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.6",
    include_package_data= True,
)
