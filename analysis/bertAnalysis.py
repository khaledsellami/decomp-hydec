import itertools
from typing import List

import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder
from transformers import AutoTokenizer, DataCollatorWithPadding, BertModel

from .abstractAnalysis import AbstractAnalysis
from .similarity import cosine_similarity


class BertAnalysis(AbstractAnalysis):
    SUPPORTED_AGGREGATION = ["mean", "sum", "combine"]

    def __init__(self, atom_tokens: List[List[str]], atoms: List, supported_atoms: List,
                 model_name: str = "bert-base-uncased", aggregation: str = "mean", embeddings=None):
        if aggregation not in self.SUPPORTED_AGGREGATION:
            raise ValueError("Unsupported aggregation method \"{}\"".format(aggregation))
        # change type of word to those without preprocessing
        self.atoms = atoms
        self.supported_atoms = supported_atoms
        self.support_map = [self.atoms.index(i) for i in self.supported_atoms]
        self.aggregation = aggregation
        self.features = atom_tokens
        self.model_name = model_name
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.model = BertModel.from_pretrained(model_name).to(self.device)
        for param in self.model.named_parameters():
            param[1].requires_grad=False
        self.embeddings = None
        if embeddings is not None:
            self.embeddings = embeddings
        elif aggregation != "combine":
            self.embeddings = self.get_embeddings(self.features)

    def get_embeddings(self, features):
        dataset = self.tokenizer(
            [" ".join(x) for x in features], truncation=True, padding=True, return_tensors="pt")
        # dataset = Dataset.from_dict({"words": features}).map(lambda x: {"sentence": " ".join(x["words"])})
        # dataset = dataset.map(lambda x: self.tokenizer(
        #     x["sentence"], truncation=True, padding=True, return_tensors="pt"), batched=True)
        #dataset = Dataset.from_dict({"words": features}).map(f, batched=True)
        # print(torch.LongTensor(dataset["input_ids"]).shape)
        output = self.model(input_ids=torch.LongTensor(dataset["input_ids"]).to(self.device),
                            attention_mask=torch.LongTensor(dataset["attention_mask"]).to(self.device),
                            return_dict=False)
        embeddings = output[0][:, 0, :].view(-1, 768).cpu().detach().numpy()
        return embeddings

    def aggregate(self, current_clusters: List[int]):
        aggregation_map = OneHotEncoder(sparse=False, dtype=int).fit_transform(
            np.array(current_clusters).reshape(-1, 1)
        )
        if self.aggregation == "mean":
            features = aggregation_map.T.dot(self.embeddings)/aggregation_map.sum(axis=0).reshape(-1,1)
        elif self.aggregation == "sum":
            features = aggregation_map.T.dot(self.embeddings)
        elif self.aggregation == "combine":
            new_words = list()
            for j in range(aggregation_map.shape[1]):
                new_words.append(list(itertools.chain(*[self.features[i] for i in np.where(aggregation_map[:, j])[0]])))
            features = self.get_embeddings(new_words)
        else:
            raise ValueError("Unsupported aggregation method \"{}\"".format(self.aggregation))
        return features, aggregation_map

    def calculate_similarity(self, features: np.ndarray):
        return cosine_similarity(features)
