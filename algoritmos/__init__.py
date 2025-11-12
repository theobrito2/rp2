"""Algoritmos de machine learning para análise de microdados de matrícula."""

from .lda import LDAAnalysis
from .neural_network import NeuralNetworkAnalysis
from .random_forest import RandomForestAnalysis

__all__ = ['RandomForestAnalysis', 'NeuralNetworkAnalysis', 'LDAAnalysis']
