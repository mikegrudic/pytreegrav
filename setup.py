import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Runtime dependencies, stated here rather than read from requirements.txt. `python -m build`
# builds the wheel from the sdist, requirements.txt was never in the sdist (there is no
# MANIFEST.in), so the file read silently produced an empty list and every release so far has
# shipped with no declared dependencies at all -- check requires_dist on PyPI for 1.1.4.
#
# requirements.txt stays as the dev/CI list; it is not the same set. healpy and pyerfa appear
# there but are imported nowhere in the package, and pytest is test-only.
#
# numba>=0.57 is a hard floor: treewalk and grouped_treewalk import parallel_chunksize at module
# scope, which was added in 0.57, so anything older fails at import rather than at use.
install_requires = [
    "numpy",
    "numba>=0.57",
    "scipy",  # scipy.spatial.transform.Rotation, used to build the ray grid in treewalk
]

# Optional CUDA backend (pytreegrav.cuda), used only when explicitly imported or via
# ColumnDensity(device="cuda"). numba-cuda provides numba.cuda from numba 0.62 on, where the CUDA
# target was split out of numba proper; it pulls in the cuda-bindings/core/pathfinder wheels and
# leaves numba, numpy and llvmlite untouched.
extras_require = {"cuda": ["numba-cuda>=0.30"]}

setuptools.setup(
    name="pytreegrav",
    version="1.2.0",
    author="Mike Grudic",
    author_email="mike.grudich@gmail.com",
    description="Fast approximate gravitational force and potential calculations",
    long_description=long_description,
    long_description_content_type="text/markdown",
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
    install_requires=install_requires,
    extras_require=extras_require,
)
