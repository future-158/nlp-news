# .prog install
NAME := rapids-22.04

install: # install rapids -> pytorch -> sentence_transformer
	conda create -y -n $(NAME) -c rapidsai -c nvidia -c conda-forge \
    cuml=22.04 python=3.8 cudatoolkit=11.0
	conda install -y -n $(NAME) -c pytorch pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0
	conda install -y -n $(NAME) -c conda-forge sentence-transformers
	pip install tensorflow_decision_forests --upgrade
	conda install -y -n $(NAME) -c conda-forge imbalanced-learn ipykernel omegaconf joblib
	
sbert:
	conda create --name sbert -y -c pytorch -c conda-forge \
	pytorch cudatoolkit=11.3 sentence-transformers \
	pip ipykernel omegaconf joblib matplotlib \
	imbalanced-learn pandas
	conda run --name sbert python -m pip install \
	tensorflow tensorflow_decision_forests --upgrade
	pip install -U transformers tokenizers

