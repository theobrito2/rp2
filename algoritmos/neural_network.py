"""Análise de Rede Neural para predição de evasão escolar usando PyTorch."""

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
import torch
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset


class DropoutDataset(Dataset):
	"""Dataset customizado para dados de evasão escolar."""

	def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
		"""
		Inicializa dataset.

		Args:
			X: Matriz de features
			y: Labels alvo
		"""
		self.X: torch.Tensor = torch.FloatTensor(X)
		self.y: torch.Tensor = torch.FloatTensor(y)

	def __len__(self) -> int:
		"""Retorna tamanho do dataset."""
		return len(self.X)

	def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
		"""Retorna item do dataset por índice."""
		return self.X[idx], self.y[idx]


class DropoutNN(nn.Module):
	"""Rede Neural para predição de evasão escolar."""

	def __init__(self, input_size: int, hidden_sizes: list[int] = [256, 128, 64], dropout_rate: float = 0.3) -> None:
		"""
		Inicializa arquitetura da rede neural.

		Args:
			input_size: Número de features de entrada
			hidden_sizes: Lista de tamanhos das camadas ocultas
			dropout_rate: Taxa de dropout para regularização
		"""
		super().__init__()

		layers: list[nn.Module] = []
		prev_size = input_size

		# Construir camadas ocultas
		for hidden_size in hidden_sizes:
			layers.append(nn.Linear(prev_size, hidden_size))
			layers.append(nn.BatchNorm1d(hidden_size))
			layers.append(nn.ReLU())
			layers.append(nn.Dropout(dropout_rate))
			prev_size = hidden_size

		# Camada de saída
		layers.append(nn.Linear(prev_size, 1))
		layers.append(nn.Sigmoid())

		self.network = nn.Sequential(*layers)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""
		Forward pass da rede.

		Args:
			x: Tensor de entrada

		Returns:
			Tensor de saída após processamento
		"""
		return self.network(x)


class NeuralNetworkAnalysis:
	"""Análise de Rede Neural para predição de evasão escolar."""

	def __init__(self, df: pd.DataFrame, eval_column: str, wanted_columns: tuple[str, ...]) -> None:
		"""
		Inicializa análise de Rede Neural.

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
		self.model: Optional[DropoutNN] = None
		self.device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.df_encoded: Optional[pd.DataFrame] = None
		self.train_losses: list[float] = []
		self.val_losses: list[float] = []
		self.train_accuracies: list[float] = []
		self.val_accuracies: list[float] = []
		self.best_model_state: Optional[dict] = None
		self.train_loader: Optional[DataLoader] = None
		self.val_loader: Optional[DataLoader] = None
		self.test_loader: Optional[DataLoader] = None
		self.test_labels: Optional[np.ndarray] = None
		self.test_predictions: Optional[np.ndarray] = None
		self.test_probabilities: Optional[np.ndarray] = None

	def preprocess_data(self) -> None:
		"""Preprocessa os dados: codificação binária, codificação de rótulos e normalização."""
		print('Preprocessando dados...')
		print(f'Dispositivo: {self.device}')

		# Converter CUDA dataframe para pandas se necessário
		df_copy = self.df.to_pandas() if self.is_using_rapids else self.df.copy()

		# Distribuição das classes antes da conversão
		print('\nDistribuição das classes (antes):')
		print(df_copy[self.eval_column].value_counts(normalize=True))

		# Converter para binário: 0 = Evadidos, 1 = Outros
		df_copy[self.eval_column] = (df_copy[self.eval_column] != 'Evadidos').astype(int)

		print('\nDistribuição das classes (depois):')
		print(df_copy[self.eval_column].value_counts(normalize=True))

		# Transformar todas as colunas categóricas para numéricas
		self.df_encoded = df_copy[[self.eval_column, *self.wanted_columns]].copy()

		for col in self.wanted_columns:
			le = LabelEncoder()
			self.df_encoded[col] = le.fit_transform(df_copy[col])
			self.encoders[col] = le

		print('\nDados preprocessados!')

	def prepare_dataloaders(self, batch_size: int = 1024, test_size: float = 0.2, val_size: float = 0.1) -> None:
		"""
		Prepara dataloaders de treino, validação e teste.

		Args:
			batch_size: Tamanho do batch para treinamento
			test_size: Proporção de dados para teste
			val_size: Proporção de dados de treino para validação
		"""
		print(f'\nPreparando dataloaders (batch_size={batch_size})...')

		X = self.df_encoded.drop(self.eval_column, axis=1).fillna(0).to_numpy()
		y = self.df_encoded[self.eval_column].to_numpy()

		# Dividir em treino+val e teste
		X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

		# Dividir treino em treino e validação
		X_train, X_val, y_train, y_val = train_test_split(
			X_temp,
			y_temp,
			test_size=val_size,
			random_state=42,
			stratify=y_temp,
		)

		# Normalizar features
		X_train = self.scaler.fit_transform(X_train)
		X_val = self.scaler.transform(X_val)
		X_test = self.scaler.transform(X_test)

		# Criar datasets
		train_dataset = DropoutDataset(X_train, y_train)
		val_dataset = DropoutDataset(X_val, y_val)
		test_dataset = DropoutDataset(X_test, y_test)

		# Criar dataloaders
		self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
		self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
		self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

		print(f'Train: {len(train_dataset)} samples')
		print(f'Validation: {len(val_dataset)} samples')
		print(f'Test: {len(test_dataset)} samples')

	def train_model(
		self,
		epochs: int = 20,
		lr: float = 0.001,
		hidden_sizes: list[int] = [256, 128, 64],
		dropout_rate: float = 0.3,
	) -> None:
		"""
		Treina a rede neural.

		Args:
			epochs: Número de épocas de treinamento
			lr: Taxa de aprendizado
			hidden_sizes: Lista de tamanhos das camadas ocultas
			dropout_rate: Taxa de dropout para regularização
		"""
		print(f'\nTreinando Neural Network ({epochs} epochs)...')

		# Inicializar modelo
		input_size = len(self.wanted_columns)
		self.model = DropoutNN(input_size, hidden_sizes, dropout_rate).to(self.device)

		# Função de perda e otimizador
		criterion = nn.BCELoss()
		optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
		scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

		# Loop de treinamento
		best_val_loss = float('inf')

		for epoch in range(epochs):
			# Fase de treinamento
			self.model.train()
			train_loss = 0.0
			train_correct = 0
			train_total = 0

			for batch_X, batch_y in self.train_loader:
				batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

				# Forward pass
				outputs = self.model(batch_X).squeeze()
				loss = criterion(outputs, batch_y)

				# Backward pass
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				# Estatísticas
				train_loss += loss.item()
				predictions = (outputs > 0.5).float()
				train_correct += (predictions == batch_y).sum().item()
				train_total += batch_y.size(0)

			train_loss /= len(self.train_loader)
			train_accuracy = train_correct / train_total

			# Fase de validação
			self.model.eval()
			val_loss = 0.0
			val_correct = 0
			val_total = 0

			with torch.no_grad():
				for batch_X, batch_y in self.val_loader:
					batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

					outputs = self.model(batch_X).squeeze()
					loss = criterion(outputs, batch_y)

					val_loss += loss.item()
					predictions = (outputs > 0.5).float()
					val_correct += (predictions == batch_y).sum().item()
					val_total += batch_y.size(0)

			val_loss /= len(self.val_loader)
			val_accuracy = val_correct / val_total

			# Salvar métricas
			self.train_losses.append(train_loss)
			self.val_losses.append(val_loss)
			self.train_accuracies.append(train_accuracy)
			self.val_accuracies.append(val_accuracy)

			# Agendamento de taxa de aprendizado
			scheduler.step(val_loss)

			# Salvar melhor modelo
			if val_loss < best_val_loss:
				best_val_loss = val_loss
				self.best_model_state = self.model.state_dict().copy()

			# Imprimir progresso
			if (epoch + 1) % 5 == 0 or epoch == 0:
				print(
					f'Epoch {epoch + 1}/{epochs} - '
					f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f} - '
					f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}',
				)

		# Carregar melhor modelo
		self.model.load_state_dict(self.best_model_state)
		print(f'\nTreinamento concluído! Melhor Val Loss: {best_val_loss:.4f}')

	def evaluate_model(self) -> None:
		"""Avalia o modelo nos dados de teste."""
		print('\nAvaliando modelo no conjunto de teste...')

		self.model.eval()
		all_predictions: list[float] = []
		all_probabilities: list[float] = []
		all_labels: list[float] = []

		with torch.no_grad():
			for batch_X, batch_y in self.test_loader:
				batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

				outputs = self.model(batch_X).squeeze()
				predictions = (outputs > 0.5).float()

				all_predictions.extend(predictions.cpu().numpy())
				all_probabilities.extend(outputs.cpu().numpy())
				all_labels.extend(batch_y.cpu().numpy())

		# Converter para arrays numpy
		self.test_labels = np.array(all_labels)
		self.test_predictions = np.array(all_predictions)
		self.test_probabilities = np.array(all_probabilities)

		# Relatório de classificação
		print('\nRelatório de Classificação:')
		print(
			classification_report(self.test_labels, self.test_predictions, target_names=['Evadido', 'Ativo/Concluído']),
		)

		# Score ROC-AUC
		roc_auc = roc_auc_score(self.test_labels, self.test_probabilities)
		print(f'\nROC-AUC Score: {roc_auc:.4f}')

	def plot_training_history(self) -> None:
		"""Plota métricas de treinamento e validação."""
		print('\nGerando gráficos de treinamento...')

		fig, axes = plt.subplots(1, 2, figsize=(14, 5))

		# Plot de perda
		axes[0].plot(self.train_losses, label='Train Loss', color='blue')
		axes[0].plot(self.val_losses, label='Validation Loss', color='red')
		axes[0].set_xlabel('Epoch')
		axes[0].set_ylabel('Loss')
		axes[0].set_title('Training and Validation Loss')
		axes[0].legend()
		axes[0].grid(True)

		# Plot de acurácia
		axes[1].plot(self.train_accuracies, label='Train Accuracy', color='blue')
		axes[1].plot(self.val_accuracies, label='Validation Accuracy', color='red')
		axes[1].set_xlabel('Epoch')
		axes[1].set_ylabel('Accuracy')
		axes[1].set_title('Training and Validation Accuracy')
		axes[1].legend()
		axes[1].grid(True)

		plt.tight_layout()
		plt.show()

	def plot_confusion_matrix(self) -> None:
		"""Plota matriz de confusão."""
		print('\nGerando matriz de confusão...')

		cm = confusion_matrix(self.test_labels, self.test_predictions)

		plt.figure(figsize=(8, 6))
		sns.heatmap(
			cm,
			annot=True,
			fmt='d',
			cmap='Blues',
			xticklabels=['Evadido', 'Ativo/Concluído'],
			yticklabels=['Evadido', 'Ativo/Concluído'],
		)
		plt.xlabel('Predição')
		plt.ylabel('Real')
		plt.title('Matriz de Confusão')
		plt.show()

	def plot_roc_curve(self) -> None:
		"""Plota curva ROC."""
		print('\nGerando curva ROC...')

		fpr, tpr, thresholds = roc_curve(self.test_labels, self.test_probabilities)
		roc_auc = roc_auc_score(self.test_labels, self.test_probabilities)

		plt.figure(figsize=(8, 6))
		plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
		plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.05])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title('Receiver Operating Characteristic (ROC) Curve')
		plt.legend(loc='lower right')
		plt.grid(True)
		plt.show()

	def save_model(self, filepath: str = 'dropout_model.pth') -> None:
		"""
		Salva o modelo treinado.

		Args:
			filepath: Caminho para salvar o modelo
		"""
		torch.save(
			{
				'model_state_dict': self.model.state_dict(),
				'scaler': self.scaler,
				'encoders': self.encoders,
				'input_size': len(self.wanted_columns),
			},
			filepath,
		)
		print(f'\nModelo salvo em {filepath}')

	def save_analysis(self, filename: str = 'neural_network_analysis.csv') -> None:
		"""
		Salva histórico de treinamento em CSV.

		Args:
			filename: Nome do arquivo para salvar
		"""
		import pandas as pd

		history_df = pd.DataFrame({
			'epoch': range(1, len(self.train_losses) + 1),
			'train_loss': self.train_losses,
			'val_loss': self.val_losses,
			'train_accuracy': self.train_accuracies,
			'val_accuracy': self.val_accuracies,
		})

		history_df.to_csv(filename, index=False)
		print(f'\nHistórico de treinamento salvo em {filename}')

	def run(self, epochs: int = 20, batch_size: int = 1024, lr: float = 0.001) -> None:
		"""
		Executa pipeline completo de análise de Rede Neural.

		Args:
			epochs: Número de épocas de treinamento
			batch_size: Tamanho do batch para treinamento
			lr: Taxa de aprendizado
		"""
		print('=' * 60)
		print('NEURAL NETWORK ANALYSIS - Predição de Evasão Escolar')
		print('=' * 60)

		self.preprocess_data()
		self.prepare_dataloaders(batch_size=batch_size)
		self.train_model(epochs=epochs, lr=lr)
		self.evaluate_model()
		self.plot_training_history()
		self.plot_confusion_matrix()
		self.plot_roc_curve()
		self.save_model()
		self.save_analysis()

		print('\n' + '=' * 60)
		print('Análise Neural Network concluída!')
		print('=' * 60)
