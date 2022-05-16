from sentence_transformers import SentenceTransformer

import pickle
import os
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from omegaconf import OmegaConf
import joblib
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from statistics import mode
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC, LinearSVC
from sentence_transformers import SentenceTransformer, LoggingHandler, losses, util
from sentence_transformers.datasets import SentenceLabelDataset
from torch.utils.data import DataLoader
from sentence_transformers.readers import InputExample
from sentence_transformers.evaluation import TripletEvaluator
from datetime import datetime
import torch

cfg = OmegaConf.load("conf/config.yaml")

df = pd.read_csv(
    cfg.catalog.translated
)


valid_index = (
    df
    [df.split!='test']
    .groupby('category')
    .sample(frac=0.2) # 80 20 split
    .index
)

valid_mask = df.index.isin(valid_index)
df.loc[valid_mask, 'split'] = 'valid'

cls2idx = {k:i for i, k in enumerate(sorted(df.category.unique()))}
labels = df.category.map(cls2idx).values

train_mask = df.split == 'train'
valid_mask = df.split == 'valid'
test_mask = df.split == 'test'

assert len(cls2idx ) == 8 # sanity test

def trec_dataset(dataframe):
    cat_sentences = list(dataframe.groupby('category'))

    triplets = []

    for category, sentences in cat_sentences:
        for tup in sentences.itertuples():
            pos_order = np.random.choice(sentences[sentences.order != tup.order].order)
            pos =  sentences.loc[sentences.order == pos_order, 'en_sentence'].values[0]

            neg_category = np.random.choice(
                [key for key in cls2idx.keys() if key != category])
            neg = dataframe[dataframe.category == neg_category].sample(1).en_sentence.values[0]
            # triplets.append([tup.en_sentence, pos, neg])
            triplet = InputExample(texts=[tup.en_sentence, pos, neg])
            triplets.append(triplet)
    return triplets
            
train_set = trec_dataset(df[train_mask])
valid_set = trec_dataset(df[valid_mask])

# We create a special dataset "SentenceLabelDataset" to wrap out train_set
# It will yield batches that contain at least two samples with the same label
train_data_sampler = SentenceLabelDataset(train_set)
train_dataloader = DataLoader(
    train_data_sampler,
    batch_size=16,
    drop_last=True
    )


model = SentenceTransformer('all-distilroberta-v1',
device='cuda'
)
model.max_seq_length = 500

# train_loss = losses.BatchAllTripletLoss(model=model)
train_loss = losses.BatchHardTripletLoss(model=model)
#train_loss = losses.BatchHardSoftMarginTripletLoss(model=model)
#train_loss = losses.BatchSemiHardTripletLoss(model=model)

valid_evaluator = TripletEvaluator.from_input_examples(valid_set, name='trec-valid')
valid_evaluator(model)

num_epochs = 1

warmup_steps = int(len(train_dataloader) * num_epochs  * 0.1)  # 10% of train data

# Train the model
output_path = (
    cfg.catalog.model.output_path
    + "-"
    + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
)

# model_parameters = filter(lambda p: p.requires_grad, model.parameters())
# params = sum([p.sum() for p in model_parameters])

# loaded = torch.load('output/finetune-batch-hard-trec--2022-05-15_23-36-10/pytorch_model.bin')
# model = model.load_state_dict(state_dict=loaded, strict=False)


model = model.train(True)
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    # evaluator=valid_evaluator,
    epochs=num_epochs,
    evaluation_steps=1000,
    # warmup_steps=warmup_steps,
    optimizer_class=torch.optim.AdamW,
    output_path=output_path,
)

# check before after
model = model.train(False)
valid_evaluator(model)

encode_kwargs = dict(
        batch_size = 128,
        # output_value = 'token_embeddings',
        convert_to_numpy = True,
        show_progress_bar=True,
        device = 'cuda',
        normalize_embeddings=True
)

embeddings = model.encode(
    df.en_sentence.values,
    **encode_kwargs
    )

# fit svc
model = LinearSVC()
model.fit(embeddings[train_mask], labels[train_mask])

valid_pred = model.predict(embeddings[valid_mask])
valid_acc = accuracy_score(
    labels[valid_mask], valid_pred
)
assert valid_acc > 0.7


# refit with train + valid
train_valid_mask = np.logical_or(train_mask, valid_mask)
model = LinearSVC()
model.fit(embeddings[train_valid_mask], labels[train_valid_mask])

test_pred = model.predict(embeddings[test_mask])
test_acc = accuracy_score(
    labels[test_mask], test_pred
)


idx2cls = {v:k for k,v in cls2idx.items()}
assert test_acc > 0.7

OmegaConf.save(
    OmegaConf.create(
    {
    'accuracy': {
        'test': float(test_acc),
    }
    }), cfg.catalog.metric.en_finetune
    )

idx2cls = {v:k for k,v in cls2idx.items()}
df.loc[test_mask, 'result'] = pd.Series(test_pred).map(idx2cls).values
df.to_csv(
        cfg.catalog.output.en_base, index=False
    )
