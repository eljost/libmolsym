from setuptools import find_packages, setup


setup(
    name="libmolsym",
    version="0.0.1",
    url="https://github.com/eljost/libmolsym",
    maintainer="Johannes Steinmetzer",
    maintainer_email="johannes.steinmetzer@uni-jena.de",
    license="License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    platforms=["unix"],
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
    ],
)
