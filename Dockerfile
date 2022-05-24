FROM huggingface/transformers-pytorch-gpu:latest as base
RUN apt update && apt install curl wget git -yq


From base
ENV port=8888
RUN curl -fsSL https://code-server.dev/install.sh | sh
RUN code-server --install-extension ms-python.python
CMD PASSWORD=ust21 code-server --bind-addr 0.0.0.0:$port

