import numpy as np
import torch
from torch.utils.data import Dataset
import os

class CustomDataset(Dataset):
	def __init__(self, features_dir, labels_dir):
		self.features_dir = features_dir
		self.labels_dir = labels_dir

		self.features_files = sorted(os.listdir(features_dir))
		self.labels_files = sorted(os.listdir(labels_dir))

	def __len__(self):
		assert len(self.features_files) == len(self.labels_files), \
				"Number of feature files and label files should be same"
		return len(self.features_files) * 32

	def __getitem__(self, idx):
		real_idx = idx // 32
		offset = idx % 32
		feature_file = self.features_files[real_idx]
		label_file = self.labels_files[real_idx]

		features = np.load(os.path.join(self.features_dir, feature_file))
		labels = np.load(os.path.join(self.labels_dir, label_file))
		return torch.from_numpy(features)[offset], torch.from_numpy(labels)[offset]