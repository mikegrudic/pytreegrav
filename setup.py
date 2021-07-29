import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pytreegrav",
    version="0.26",
    author="Mike Grudic",
    author_email="mike.grudich@gmail.com",
    description='Fast approximate gravitational force and potential calculations',
    long_description=long_description,
    long_description_content_type='text/markdown',
    project_urls={
        "Bug Tracker": "https://github.com/mikegrudic/pytreegrav",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)
