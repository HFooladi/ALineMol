# Overview

`ALineMol` is a Python library designed to assist in drug discovery by providing powerful methods for estimating the out-of-distribution (OOD) performance of molecular machine learning models, including both classical machine learning and graph neural networks (GNNs). Built on top of the popular PyTorch library, `ALineMol` offers a simple, user-friendly API for assessing OOD performance. It is designed to be flexible and easy to integrate into existing workflows.

The library first generates OOD data based on various splitting strategies, then benchmarks and evaluates the performance of different models on this OOD data. This approach helps estimate the generalization power and robustness of models to OOD data.
