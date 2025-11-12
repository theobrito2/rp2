"""Análise de Linear Discriminant Analysis para predição de evasão escolar."""

from typing import Optional

try:
	import cudf as pd
except ImportError:
	import pandas as pd

try:
	import cupy as np
except ImportError:
	import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
	accuracy_score,
	classification_report,
	confusion_matrix,
	f1_score,
	precision_score,
	recall_score,
	roc_auc_score,
	roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


class LDAAnalysis:
	"""Análise de Linear Discriminant Analysis para predição de evasão escolar."""

	def __init__(self, df: pd.DataFrame, eval_column: str, wanted_columns: tuple[str, ...]) -> None:
		"""
		Inicializa análise de LDA.

		Args:
			df: DataFrame de entrada com dados de matrícula de estudantes
			eval_column: Coluna alvo para predição
			wanted_columns: Colunas de features para usar na predição
		"""
		self.df: pd.DataFrame = df
		self.eval_column: str = eval_column
		self.wanted_columns: tuple[str, ...] = wanted_columns
		self.is_using_rapids: bool = 'cudf' in pd.__name__
		self.encoders: dict[str, LabelEncoder] = {}
		self.scaler: StandardScaler = StandardScaler()
		self.model: Optional[LinearDiscriminantAnalysis] = None
		self.df_processed: Optional[pd.DataFrame] = None
		self.df_encoded: Optional[pd.DataFrame] = None
		# Dados de teste para avaliação
		self.X_test: Optional[np.ndarray] = None
		self.y_test: Optional[np.ndarray] = None
		self.y_pred: Optional[np.ndarray] = None
		self.y_proba: Optional[np.ndarray] = None
		# Métricas de avaliação
		self.accuracy: Optional[float] = None
		self.precision: Optional[float] = None
		self.recall: Optional[float] = None
		self.f1: Optional[float] = None
		self.roc_auc: Optional[float] = None
		self.confusion_mat: Optional[np.ndarray] = None

	def preprocess_data(self) -> None:
		"""Preprocessa os dados: codificação binária, codificação de rótulos e normalização."""
		print('Preprocessando dados...')

		# Converter CUDA dataframe para pandas se necessário
		df_copy = self.df.to_pandas() if self.is_using_rapids else self.df.copy()

		# Distribuição das classes antes da conversão
		print('\nDistribuição das classes (antes):')
		print(df_copy[self.eval_column].value_counts(normalize=True))

		# Converter para binário: 0 = Evadidos, 1 = Outros
		# Verificar se a coluna já está em formato numérico ou é string
		if df_copy[self.eval_column].dtype == 'object':
			# Coluna contém strings, fazer conversão normal
			df_copy[self.eval_column] = (df_copy[self.eval_column] != 'Evadidos').astype(int)
		else:
			# Coluna já é numérica, assumir que 0 = Evadidos, 1 = Outros
			print('(Coluna já está em formato numérico)')

		print('\nDistribuição das classes (depois):')
		print(df_copy[self.eval_column].value_counts(normalize=True))

		# Armazenar cópia processada para uso em outros métodos
		self.df_processed = df_copy

		# Transformar todas as colunas categóricas para numéricas
		self.df_encoded = df_copy[[self.eval_column, *self.wanted_columns]].copy()

		for col in self.wanted_columns:
			le = LabelEncoder()
			self.df_encoded[col] = le.fit_transform(df_copy[col])
			self.encoders[col] = le

		print('\nDados preprocessados!')

	def train_model(self, test_size: float = 0.2, solver: str = 'svd') -> None:
		"""
		Treina o modelo LDA.

		Args:
			test_size: Proporção de dados para teste
			solver: Solver para o LDA ('svd', 'lsqr', 'eigen')
		"""
		print(f'\nTreinando modelo LDA (solver={solver})...')

		X = self.df_encoded.drop(self.eval_column, axis=1).fillna(0).to_numpy()
		y = self.df_encoded[self.eval_column].to_numpy()

		# Dividir os dados em treino e teste
		X_train, X_test, y_train, y_test = train_test_split(
			X,
			y,
			test_size=test_size,
			random_state=42,
			stratify=y,
		)

		# Normalizar features (importante para LDA)
		X_train = self.scaler.fit_transform(X_train)
		X_test = self.scaler.transform(X_test)

		# Armazenar dados de teste
		self.X_test = X_test
		self.y_test = y_test

		# Criar e treinar modelo LDA
		self.model = LinearDiscriminantAnalysis(solver=solver)
		self.model.fit(X_train, y_train)

		# Fazer predições
		self.y_pred = self.model.predict(X_test)
		self.y_proba = self.model.predict_proba(X_test)[:, 1]

		print('Treinamento concluído!')

	def evaluate_model(self) -> None:
		"""Avalia o modelo nos dados de teste."""
		print('\nAvaliando modelo no conjunto de teste...')

		# Calcular métricas
		self.accuracy = accuracy_score(self.y_test, self.y_pred)
		self.precision = precision_score(self.y_test, self.y_pred, average='binary', zero_division=0)
		self.recall = recall_score(self.y_test, self.y_pred, average='binary', zero_division=0)
		self.f1 = f1_score(self.y_test, self.y_pred, average='binary', zero_division=0)
		self.roc_auc = roc_auc_score(self.y_test, self.y_proba)
		self.confusion_mat = confusion_matrix(self.y_test, self.y_pred)

		print('\n' + '=' * 60)
		print('MÉTRICAS DE AVALIAÇÃO')
		print('=' * 60)
		print(f'Accuracy:  {self.accuracy:.4f}')
		print(f'Precision: {self.precision:.4f}')
		print(f'Recall:    {self.recall:.4f}')
		print(f'F1 Score:  {self.f1:.4f}')
		print(f'ROC-AUC:   {self.roc_auc:.4f}')
		print('=' * 60)

		# Relatório de classificação detalhado
		print('\nRelatório de Classificação Detalhado:')
		print(
			classification_report(self.y_test, self.y_pred, target_names=['Evadido', 'Ativo/Concluído']),
		)

		# Informações sobre o modelo
		print('\nInformações do Modelo LDA:')
		print(f'Número de componentes: {self.model.n_components}')
		if hasattr(self.model, 'explained_variance_ratio_'):
			print(f'Variância explicada: {self.model.explained_variance_ratio_[0]:.4f}')

	def plot_confusion_matrix(self) -> None:
		"""Plota matriz de confusão."""
		print('\nGerando matriz de confusão...')

		plt.figure(figsize=(8, 6))
		sns.heatmap(
			self.confusion_mat,
			annot=True,
			fmt='d',
			cmap='Blues',
			xticklabels=['Evadido', 'Ativo/Concluído'],
			yticklabels=['Evadido', 'Ativo/Concluído'],
		)
		plt.xlabel('Predição')
		plt.ylabel('Real')
		plt.title('Matriz de Confusão - LDA')
		plt.show()

	def plot_roc_curve(self) -> None:
		"""Plota curva ROC."""
		print('\nGerando curva ROC...')

		fpr, tpr, thresholds = roc_curve(self.y_test, self.y_proba)

		plt.figure(figsize=(8, 6))
		plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {self.roc_auc:.4f})')
		plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.05])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title('Receiver Operating Characteristic (ROC) Curve - LDA')
		plt.legend(loc='lower right')
		plt.grid(True)
		plt.show()

	def plot_discriminant_scores(self) -> None:
		"""Plota distribuição dos scores discriminantes."""
		print('\nGerando distribuição de scores discriminantes...')

		# Calcular scores discriminantes
		scores = self.model.decision_function(self.X_test)

		plt.figure(figsize=(10, 6))

		# Plotar histogramas separados por classe
		plt.hist(
			scores[self.y_test == 0],
			bins=50,
			alpha=0.5,
			label='Evadido',
			color='red',
			density=True,
		)
		plt.hist(
			scores[self.y_test == 1],
			bins=50,
			alpha=0.5,
			label='Ativo/Concluído',
			color='blue',
			density=True,
		)

		plt.axvline(x=0, color='black', linestyle='--', label='Limite de Decisão')
		plt.xlabel('Score Discriminante')
		plt.ylabel('Densidade')
		plt.title('Distribuição de Scores Discriminantes - LDA')
		plt.legend()
		plt.grid(True, alpha=0.3)
		plt.show()

	def save_analysis(self, metrics_filename: str = 'lda_metrics.csv') -> None:
		"""
		Salva métricas em CSV.

		Args:
			metrics_filename: Nome do arquivo para salvar métricas
		"""
		import pandas as pd

		# Salvar métricas de avaliação
		metrics_df = pd.DataFrame({
			'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC'],
			'Value': [self.accuracy, self.precision, self.recall, self.f1, self.roc_auc],
		})
		metrics_df.to_csv(metrics_filename, index=False)
		print(f'\nMétricas de avaliação salvas em {metrics_filename}')

	def run(self, test_size: float = 0.2, solver: str = 'svd') -> None:
		"""
		Executa pipeline completo de análise de LDA.

		Args:
			test_size: Proporção de dados para teste
			solver: Solver para o LDA ('svd', 'lsqr', 'eigen')
		"""
		print('=' * 60)
		print('LINEAR DISCRIMINANT ANALYSIS - Predição de Evasão Escolar')
		print('=' * 60)

		self.preprocess_data()
		self.train_model(test_size=test_size, solver=solver)
		self.evaluate_model()
		self.plot_confusion_matrix()
		self.plot_roc_curve()
		self.plot_discriminant_scores()
		self.save_analysis()

		print('\n' + '=' * 60)
		print('Análise LDA concluída!')
		print('=' * 60)
