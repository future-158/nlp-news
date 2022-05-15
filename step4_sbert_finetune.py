from sentence_transformers import SentenceTransformer
import pickle
import os
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from omegaconf import OmegaConf
import joblib
from sentence_transformers import SentenceTransformer, models
from torch import nn

cfg = OmegaConf.load("conf/config.yaml")
# load train, test
train_valid = pd.read_csv( 
    cfg.catalog.raw.train
)

test = pd.read_csv( 
    cfg.catalog.raw.test
)

# title, cleanBody, category
train_valid.category.value_counts()
test.category.value_counts()

# train test valid split
train = train_valid.groupby("category").sample(frac=0.8, random_state=1)
valid_mask = ~train_valid.index.isin(train.index)
valid = train_valid[valid_mask]

train['mergeBody'] = train.title.str.strip() + ' ' +  train.cleanBody.str.strip()
valid['mergeBody'] = valid.title.str.strip() + ' ' +  valid.cleanBody.str.strip()
test['mergeBody'] = test.title.str.strip() + ' ' +  test.cleanBody.str.strip()


from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('distiluse-base-multilingual-cased-v1', device='cuda')
# model = SentenceTransformer('distilbert-base-nli-mean-tokens', device='cuda')
model.max_seq_length = 500



encode_kwargs = dict(
        batch_size = 32,
        # output_value = 'token_embeddings',
        convert_to_numpy = True,
        # convert_to_tensor = False,
        show_progress_bar=True,
        device = 'cuda',
        normalize_embeddings=True

)

train_embeddings = model.encode(
    train.mergeBody.values,
    **encode_kwargs
    )
valid_embeddings = model.encode(
    valid.mergeBody.values,
    **encode_kwargs
    )
test_embeddings = model.encode(
    test.mergeBody.values,
    **encode_kwargs
    )


model = SentenceTransformer('distiluse-base-multilingual-cased-v1', device='cuda')
# model = SentenceTransformer('distilbert-base-nli-mean-tokens', device='cuda')
model.max_seq_length = 500


# word_embedding_model = models.Transformer('bert-base-uncased', max_seq_length=256)
# pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
# dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), out_features=256, activation_function=nn.Tanh())
# model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model])


from sentence_transformers import SentenceTransformer, InputExample
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from sentence_transformers import evaluation

train_examples = [
    InputExample(texts=['My first sentence', 'My second sentence'], label=0.8),
    InputExample(texts=['Another pair', 'Unrelated sentence'], label=0.3)
    ]
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.CosineSimilarityLoss(model)

model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100, device='cuda')

train_index = np.arange(len(train))
pd.DataFrame(index=)


sentences1 = ['This list contains the first column', 'With your sentences', 'You want your model to evaluate on']
sentences2 = ['Sentences contains the other column', 'The evaluator matches sentences1[i] with sentences2[i]', 'Compute the cosine similarity and compares it to scores[i]']
scores = [0.3, 0.6, 0.2]

evaluator = evaluation.EmbeddingSimilarityEvaluator(sentences1, sentences2, scores)

# ... Your other code to load training data

model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100, evaluator=evaluator, evaluation_steps=500)








# from some blog

# Set up data for fine-tuning 
sentence_reader = LabelSentenceReader(folder='~/tsv_files')
data_list = sentence_reader.get_examples(filename='recipe_bot_data.tsv')
triplets = triplets_from_labeled_dataset(input_examples=data_list)
finetune_data = SentencesDataset(examples=triplets, model=model)
finetune_dataloader = DataLoader(finetune_data, shuffle=True, batch_size=16)

# Initialize triplet loss
loss = TripletLoss(model=model)

# Fine-tune the model
model.fit(train_objectives=[(finetune_dataloader, loss)], epochs=4,output_path='bert-base-nli-stsb-mean-tokens-recipes')


# scipy==1.5.4, numpy==1.19.5
from scipy import spatial
import numpy as np

q_a_mappings = {'Question Embedding': [[ ], [ ], [ ], …], 'Question Text': ['What should I cook after work?', 'What are some one-pot meals I can cook?', ...], 'Corresponding Answer': ['For easy one-pot or weeknight recipes, please access this [link].', 'For easy one-pot or weeknight recipes, please access this [link].', ...]}

question_embeddings = q_a_mappings['Question Embedding']
question_texts = q_a_mappings['Question Text']
answer_mappings = q_a_mappings['Corresponding Answer']

distances = spatial.distance.cdist(np.array(encoded_question), question_embeddings, 'cosine')[0]
results = zip(range(len(distances)), distances)
results = sorted(results, key=lambda x: x[1])

for idx, distance in results[0:2]: # just getting top 2
    print(f"\nMatch {idx+1}:")
    print(question_texts[idx])
    print(answer_mappings[idx])



