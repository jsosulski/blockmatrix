# blockmatrix

A python package to provide easier working with block-structured matrices. Currently, this
code mostly serves my purposes, i.e., manipulating block-structured covariance matrices
and applying high-dimensional estimation techniques to them.

This package is also available on PyPi.

## Usage

As of now unfortunately only the code and the docstrings are available as documentation.

Running the `examples/main_spatiotemporal_manipulations.py` showcases some of the
functionality and visualizations.

## Todos

- [ ] Documentation
- [ ] Testing
- [x] Implementation of sklearn style covariance estimators
  - Moved to ToeplitzLDA package
- [x] Abstract mne channels away
  - Using optional mne dependency
- [x] Reduce unnecessary dependencies
  - `toeplitz` is now optional
