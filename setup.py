import setuptools

###########################################################################
#   Laboratory of Computational Physics // University of Padua, AY 2021/22
#   Group 2202 / Barone Nagaro Ninni Valentini
#
#  This python module implements the HAVOK analysis.
#
#  coder: F. Barone + Nagaro, Ninni, Valentini
#  last edit: 2 Jun 2022
#--------------------------------------------------------------------------
#  Open Access licence
#--------------------------------------------------------------------------

description="A library for HAVOK & reservoir learning analysis, built for a University project."

setuptools.setup(
    name="rhavok",
    version="0.1.3",
    author="F. Barone",
    description=description,
    url="https://github.com/baronefr/rhavok-analysis",
    packages=setuptools.find_packages(),
    install_requires=['gym', 'matplotlib', 'numpy'],
    python_requires='>=3',
    license='Open Access',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Scientists",
        "Topic :: Physics :: Dynamical systems",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
