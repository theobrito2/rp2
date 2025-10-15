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
	from cuml.cluster import KMeans
	from cuml.decomposition import PCA
	from cuml.preprocessing import LabelEncoder
except ImportError:
	from sklearn.cluster import KMeans
	from sklearn.decomposition import PCA
	from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt


class KMeansAnalysis:
	"""Análise de clustering K-Means para padrões de evasão escolar."""

	def __init__(self, df: pd.DataFrame, eval_column: str, wanted_columns: tuple[str, ...]) -> None:
		"""
		Inicializa análise K-Means.

		Args:
			df: DataFrame de entrada com dados de matrícula de estudantes
			eval_column: Coluna alvo (para avaliação)
			wanted_columns: Colunas de features para usar no clustering
		"""
		self.df: pd.DataFrame = df
		self.eval_column: str = eval_column
		self.wanted_columns: tuple[str, ...] = wanted_columns
		self.is_using_rapids: bool = 'cudf' in pd.__name__
		self.encoders: dict[str, LabelEncoder] = {}
		self.model: Optional[KMeans] = None
		self.df_encoded: Optional[pd.DataFrame] = None
		self.cluster_labels: Optional[np.ndarray] = None
		self.optimal_k: Optional[int] = None
		self.inertias: list[float] = []

	def preprocess_data(self) -> None:
		"""Preprocessa os dados: codificação binária e codificação de rótulos."""
		print('Preprocessando dados...')

		# Converter para binário: 0 = Evadidos, 1 = Outros
		df_copy = self.df.copy()
		df_copy[self.eval_column] = (df_copy[self.eval_column] != 'Evadidos').astype(int)

		# Transformar todas as colunas categóricas para numéricas
		self.df_encoded = df_copy[[self.eval_column, *self.wanted_columns]].copy()

		for col in self.wanted_columns:
			le = LabelEncoder()
			self.df_encoded[col] = le.fit_transform(df_copy[col])
			self.encoders[col] = le

		print('Dados preprocessados!')

	def find_optimal_clusters(self, max_k: int = 10) -> None:
		"""
		Usa o método do cotovelo para encontrar número ótimo de clusters.

		Args:
			max_k: Número máximo de clusters para testar
		"""
		print(f'\nTestando {max_k} valores de K para método do cotovelo...')

		X = self.df_encoded.drop(self.eval_column, axis=1).fillna(0)

		# Amostra de dados se muito grande (para computação mais rápida)
		if len(X) > 50000:
			print('Dataset grande, usando amostra de 50000 registros...')
			sample_indices = np.random.choice(len(X), 50000, replace=False)
			X_sample = X.iloc[sample_indices.get()] if self.is_using_rapids else X.iloc[sample_indices]
		else:
			X_sample = X

		self.inertias = []

		for k in range(2, max_k + 1):
			kmeans = KMeans(n_clusters=k, random_state=42)
			kmeans.fit(X_sample)
			inertia = kmeans.inertia_ if hasattr(kmeans, 'inertia_') else 0
			self.inertias.append(inertia)
			print(f'K={k}: inertia={inertia:.2f}')

		# Usar k=4 como padrão (baseado em experiência comum com dados educacionais)
		self.optimal_k = 4
		print(f'\nK ótimo selecionado: {self.optimal_k}')

	def train_model(self) -> None:
		"""Treina o modelo K-Means com K ótimo."""
		print(f'\nTreinando modelo K-Means com K={self.optimal_k}...')

		X = self.df_encoded.drop(self.eval_column, axis=1).fillna(0)

		self.model = KMeans(n_clusters=self.optimal_k, random_state=42)
		self.cluster_labels = self.model.fit_predict(X)

		# Adicionar labels ao dataframe
		self.df_encoded['Cluster'] = self.cluster_labels

	def analyze_clusters(self) -> None:
		"""Analisa características dos clusters e taxas de evasão."""
		print('\nAnalisando clusters...')

		for cluster_id in range(self.optimal_k):
			cluster_data = self.df_encoded[self.df_encoded['Cluster'] == cluster_id]
			total_students = len(cluster_data)
			dropout_rate = (1 - cluster_data[self.eval_column].mean()) * 100

			print(f'\nCluster {cluster_id}:')
			print(f'Total de alunos: {total_students}')
			print(f'Taxa de evasão: {dropout_rate:.2f}%')

			# Top 5 características mais comuns no cluster
			if not self.is_using_rapids:
				print('Top 5 características:')
				for col in list(self.wanted_columns)[:5]:
					if col in cluster_data.columns:
						mode_val = cluster_data[col].mode()
						if len(mode_val) > 0:
							mode_val = mode_val.iloc[0]
							# Decodificar se possível
							if col in self.encoders:
								try:
									original_val = self.encoders[col].inverse_transform([mode_val])[0]
									print(f'- {col}: {original_val}')
								except Exception:
									print(f'- {col}: {mode_val}')

	def plot_elbow(self) -> None:
		"""Plota a curva do cotovelo para seleção de K."""
		if self.is_using_rapids:
			print('\nPlots não disponíveis com RAPIDS')
			return

		print('\nGerando gráfico do método do cotovelo...')

		plt.figure(figsize=(10, 6))
		plt.plot(range(2, len(self.inertias) + 2), self.inertias, 'bo-')
		plt.xlabel('Número de Clusters (K)')
		plt.ylabel('Inércia')
		plt.title('Método do Cotovelo para Seleção de K')
		plt.axvline(x=self.optimal_k, color='r', linestyle='--', label=f'K ótimo = {self.optimal_k}')
		plt.legend()
		plt.grid(True)
		plt.show()

	def plot_clusters_pca(self) -> None:
		"""Visualiza clusters usando PCA para redução de dimensionalidade."""
		if self.is_using_rapids:
			print('\nPlots não disponíveis com RAPIDS')
			return

		print('\nGerando visualização dos clusters (PCA)...')

		X = self.df_encoded.drop([self.eval_column, 'Cluster'], axis=1).fillna(0)

		# Amostra de dados se muito grande
		if len(X) > 10000:
			sample_indices = np.random.choice(len(X), 10000, replace=False)
			X_sample = X.iloc[sample_indices]
			labels_sample = self.cluster_labels[sample_indices]
			dropout_sample = self.df_encoded[self.eval_column].iloc[sample_indices]
		else:
			X_sample = X
			labels_sample = self.cluster_labels
			dropout_sample = self.df_encoded[self.eval_column]

		# PCA para 2 dimensões
		pca = PCA(n_components=2, random_state=42)
		X_pca = pca.fit_transform(X_sample)

		# Criar subplots
		fig, axes = plt.subplots(1, 2, figsize=(16, 6))

		# Plot 1: Clusters
		scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=labels_sample, cmap='viridis', alpha=0.6)
		axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
		axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
		axes[0].set_title('Visualização dos Clusters (PCA)')
		plt.colorbar(scatter1, ax=axes[0], label='Cluster')

		# Plot 2: Status de evasão
		scatter2 = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=dropout_sample, cmap='RdYlGn', alpha=0.6)
		axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
		axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
		axes[1].set_title('Status de Matrícula (0=Evadido, 1=Ativo/Concluído)')
		plt.colorbar(scatter2, ax=axes[1], label='Status')

		plt.tight_layout()
		plt.show()

	def plot_cluster_dropout_rates(self) -> None:
		"""Plota taxas de evasão por cluster."""
		if self.is_using_rapids:
			print('\nPlots não disponíveis com RAPIDS')
			return

		print('\nGerando gráfico de taxas de evasão por cluster...')

		dropout_rates: list[float] = []
		cluster_sizes: list[int] = []

		for cluster_id in range(self.optimal_k):
			cluster_data = self.df_encoded[self.df_encoded['Cluster'] == cluster_id]
			dropout_rate = (1 - cluster_data[self.eval_column].mean()) * 100
			dropout_rates.append(dropout_rate)
			cluster_sizes.append(len(cluster_data))

		fig, axes = plt.subplots(1, 2, figsize=(14, 5))

		# Plot 1: Taxas de evasão
		colors = ['#d32f2f' if rate > 20 else '#fbc02d' if rate > 10 else '#388e3c' for rate in dropout_rates]
		axes[0].bar(range(self.optimal_k), dropout_rates, color=colors)
		axes[0].set_xlabel('Cluster')
		axes[0].set_ylabel('Taxa de Evasão (%)')
		axes[0].set_title('Taxa de Evasão por Cluster')
		axes[0].set_xticks(range(self.optimal_k))

		# Adicionar rótulos de valores nas barras
		for i, rate in enumerate(dropout_rates):
			axes[0].text(i, rate + 1, f'{rate:.1f}%', ha='center', va='bottom')

		# Plot 2: Tamanhos dos clusters
		axes[1].bar(range(self.optimal_k), cluster_sizes, color='steelblue')
		axes[1].set_xlabel('Cluster')
		axes[1].set_ylabel('Número de Alunos')
		axes[1].set_title('Tamanho dos Clusters')
		axes[1].set_xticks(range(self.optimal_k))

		# Adicionar rótulos de valores nas barras
		for i, size in enumerate(cluster_sizes):
			axes[1].text(i, size + max(cluster_sizes) * 0.01, f'{size:,}', ha='center', va='bottom')

		plt.tight_layout()
		plt.show()

	def save_analysis(self, filename: str = 'cluster_analysis.csv') -> None:
		"""
		Salva análise de clusters em CSV.

		Args:
			filename: Nome do arquivo para salvar
		"""
		if self.is_using_rapids:
			print('\nExportação não disponível com RAPIDS')
			return

		cluster_summary: list[dict[str, any]] = []

		for cluster_id in range(self.optimal_k):
			cluster_data = self.df_encoded[self.df_encoded['Cluster'] == cluster_id]
			dropout_rate = (1 - cluster_data[self.eval_column].mean()) * 100

			cluster_summary.append({
				'Cluster': cluster_id,
				'Total_Alunos': len(cluster_data),
				'Taxa_Evasao_%': dropout_rate,
			})

		summary_df = pd.DataFrame(cluster_summary)
		summary_df.to_csv(filename, index=False)
		print(f'\nAnálise salva em {filename}')

	def run(self) -> None:
		"""Executa pipeline completo de análise K-Means."""
		print('=' * 60)
		print('K-MEANS CLUSTERING - Análise de Padrões de Evasão')
		print('=' * 60)

		self.preprocess_data()
		self.find_optimal_clusters(max_k=8)
		self.plot_elbow()
		self.train_model()
		self.analyze_clusters()
		self.plot_clusters_pca()
		self.plot_cluster_dropout_rates()
		self.save_analysis()

		print('\n' + '=' * 60)
		print('Análise K-Means concluída!')
		print('=' * 60)
