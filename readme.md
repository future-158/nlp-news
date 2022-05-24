# update
05-25
add task/fine-tune step
summary
- from roberta pretrained, domain apdation with masked finetuning
- from above model, fine tune sentence transformer with CosineSimilarityLoss
- *!todo evaluation step*



# flow
- google translate로 이용하여 kr -> en 변환
- 유명한 news classficiation 데이터셋인 bbc news dataset과 카테고리가 비슷하므로,
bbc news dataset을 추가하여 학습함
- word embedding model로 sentence_transformer 사용함
- sentence_transformer finetuning 진행 중

# class imbalance
일부 라벨 개수가 매우 적음.
sentence_transformer로 embedding 한 후 smote 방법 적용 결과 성능이 하락함

# augmentation
bbc dataset 추가 후 성능이 더 떨어짐

# fine tuning
진행 중 

