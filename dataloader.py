import os
import torch

# from torchvision.io import decode_image
from torch.utils.data import Dataset


class CustomImageDataset(Dataset):
    def __init__(self, data_file):
        self.words = []
        self.labels = []
        self.labels_one_hot_encoded = []

        counter = 0
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if len(line) > 0:
                    _, label = line.split()
                    self.words.append(counter)
                    self.labels.append(label)
                    counter += 1

        class2idx = {word: idx for idx, word in enumerate(set(self.labels))}
        self.n_classes = len(class2idx.keys())
        self.n_embeddings = len(self.words)

        for val in self.labels:
            label = torch.zeros(self.n_classes)
            label[class2idx[val]] = 1.0
            self.labels_one_hot_encoded.append(label)

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        return self.words[idx], self.labels_one_hot_encoded[idx]
