"""Análise de Random Forest para predição de evasão escolar."""

from typing import Optional

try:
	import cudf as pd
except ImportError:
	import pandas as pd

try:
	import cupy as np
except ImportError:
	import numpy as np

try:
	from cuml.ensemble import RandomForestClassifier
	from cuml.model_selection import train_test_split
	from cuml.preprocessing import LabelEncoder
except ImportError:
	from sklearn.ensemble import RandomForestClassifier
	from sklearn.model_selection import train_test_split
	from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.metrics import (
	accuracy_score,
	classification_report,
	confusion_matrix,
	f1_score,
	precision_score,
	recall_score,
)


class RandomForestAnalysis:
	"""Análise de Random Forest para predição de evasão escolar."""

	def __init__(self, df: pd.DataFrame, eval_column: str, wanted_columns: tuple[str, ...]) -> None:
		"""
		Inicializa análise de Random Forest.

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
		self.model: Optional[RandomForestClassifier] = None
		self.df_processed: Optional[pd.DataFrame] = None
		self.df_encoded: Optional[pd.DataFrame] = None
		self.feature_importance: Optional[pd.Series] = None
		self.sorted_corr: Optional[pd.Series] = None
		# Métricas de avaliação
		self.accuracy: Optional[float] = None
		self.precision: Optional[float] = None
		self.recall: Optional[float] = None
		self.f1: Optional[float] = None
		self.confusion_mat: Optional[np.ndarray] = None

	def preprocess_data(self) -> None:
		"""Preprocessa os dados: codificação binária e codificação de rótulos."""
		print('Preprocessando dados...')

		# Criar cópia para evitar modificar o dataframe original
		df_copy = self.df.copy()

		# Distribuição das classes antes da conversão
		print('\nDistribuição das classes (antes):')
		print(df_copy[self.eval_column].value_counts(normalize=True))

		# Converter para binário: 0 = Evadidos, 1 = Outros
		df_copy[self.eval_column] = (df_copy[self.eval_column] != 'Evadidos').astype(int)

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

	def calculate_correlations(self) -> None:
		"""Calcula correlações de V de Cramér entre features e alvo."""
		print('\nCalculando correlações...')

		def cramers(x: pd.Series, y: pd.Series) -> float:
			"""
			Calcula a associação de duas variáveis categóricas.

			https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V#Calculation

			Args:
				x: Série de dados da primeira variável categórica
				y: Série de dados da segunda variável categórica

			Returns:
				Correlação entre as variáveis, variando de 0 (sem associação) a 1 (associação perfeita).
			"""
			confusion_matrix = pd.crosstab(x, y)

			chi2 = chi2_contingency(confusion_matrix.to_numpy())[0]
			n = confusion_matrix.sum().sum()
			min_dim = min(confusion_matrix.shape) - 1

			# Caso uma das variáveis seja constante
			if min_dim == 0:
				return 0.0

			return (
				np.sqrt(chi2 / (n * min_dim)).get().item(0) if self.is_using_rapids else np.sqrt(chi2 / (n * min_dim))
			)

		# Calculando a correlação entre todas as colunas e a coluna de avaliação
		correlations: dict[str, float] = {}

		for col in self.wanted_columns:
			correlations[col] = cramers(self.df_processed[col], self.df_processed[self.eval_column])

		self.sorted_corr = pd.Series(correlations).sort_values(ascending=False)

		print('\nTop 10 correlações:')
		print(self.sorted_corr.head(10))

	def train_model(self) -> None:
		"""Treina o modelo Random Forest."""
		print('\nTreinando modelo Random Forest...')

		X = self.df_encoded.drop(self.eval_column, axis=1).fillna(0)  # Features
		y = self.df_encoded[self.eval_column]  # Target

		# Dividir os dados em treino e teste
		X_train, X_test, y_train, y_test = train_test_split(
			X,
			y,
			test_size=0.2,
			random_state=42,
			stratify=y,
		)

		self.model = RandomForestClassifier(n_estimators=100, random_state=42)
		self.model.fit(X_train, y_train)

		y_pred = self.model.predict(X_test)

		# Calcular métricas
		self.accuracy = accuracy_score(y_test, y_pred)
		self.precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
		self.recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
		self.f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
		self.confusion_mat = confusion_matrix(y_test, y_pred)

		print('\n' + '=' * 60)
		print('MÉTRICAS DE AVALIAÇÃO')
		print('=' * 60)
		print(f'Accuracy:  {self.accuracy:.4f}')
		print(f'Precision: {self.precision:.4f}')
		print(f'Recall:    {self.recall:.4f}')
		print(f'F1 Score:  {self.f1:.4f}')
		print('=' * 60)

		print('\nRelatório de Classificação Detalhado:')
		print(classification_report(y_test, y_pred, target_names=['Evadido', 'Ativo/Concluído']))

		# Calcular importância das features
		if not self.is_using_rapids:
			self.feature_importance = pd.Series(
				self.model.feature_importances_,
				index=X.columns,
			).sort_values(ascending=False)

			print('\nTop 10 Features Mais Importantes:')
			print(self.feature_importance.head(10))

	def plot_confusion_matrix(self) -> None:
		"""Plota matriz de confusão."""
		if self.is_using_rapids:
			print('\nPlots não disponíveis com RAPIDS')
			return

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
		plt.title('Matriz de Confusão - Random Forest')
		plt.show()

	def plot_feature_analysis(self) -> None:
		"""Plota análise de importância de features vs correlação."""
		if self.is_using_rapids:
			print('\nPlots não disponíveis com RAPIDS')
			return

		print('\nGerando visualização de features...')

		# Plotando a importância das features x correlação para colunas com correlação > 0.2
		important_features = self.sorted_corr[self.sorted_corr > 0.2].index.tolist()

		plt.figure(figsize=(12, 6))
		sns.scatterplot(
			x=self.feature_importance[important_features],
			y=self.sorted_corr[important_features],
		)

		# Adicionando anotações para cada ponto
		for feature in important_features:
			plt.text(
				self.feature_importance[feature],
				self.sorted_corr[feature],
				feature,
				fontsize=9,
				ha='center',
				va='top',
				rotation=15,
				rotation_mode='anchor',
			)

		plt.xlabel('Importância da Feature (Random Forest)')
		plt.ylabel('Correlação (V de Cramér)')
		plt.title('Importância das Features vs Correlação com a Avaliação')
		plt.grid(True)
		plt.show()

	def save_analysis(self, filename: str = 'feature_analysis.csv', metrics_filename: str = 'random_forest_metrics.csv') -> None:
		"""
		Salva análise de features e métricas em CSV.

		Args:
			filename: Nome do arquivo para salvar análise de features
			metrics_filename: Nome do arquivo para salvar métricas
		"""
		if self.is_using_rapids:
			print('\nExportação não disponível com RAPIDS')
			return

		# Salvar análise de features
		feature_analysis = pd.DataFrame({
			'Feature Importance': self.feature_importance,
			"Cramér's V Correlation": self.sorted_corr,
		})
		feature_analysis.to_csv(filename)
		print(f'\nAnálise de features salva em {filename}')

		# Salvar métricas de avaliação
		metrics_df = pd.DataFrame({
			'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
			'Value': [self.accuracy, self.precision, self.recall, self.f1],
		})
		metrics_df.to_csv(metrics_filename, index=False)
		print(f'Métricas de avaliação salvas em {metrics_filename}')

	def run(self) -> None:
		"""Executa pipeline completo de análise Random Forest."""
		print('=' * 60)
		print('RANDOM FOREST ANALYSIS - Predição de Evasão Escolar')
		print('=' * 60)

		self.preprocess_data()
		self.calculate_correlations()
		self.train_model()
		self.plot_confusion_matrix()
		self.plot_feature_analysis()
		self.save_analysis()

		print('\n' + '=' * 60)
		print('Análise Random Forest concluída!')
		print('=' * 60)
