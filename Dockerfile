FROM python:3.9-slim

COPY . /app
WORKDIR  /app

# Environment variable
ENV PORT 8080

# Install required packages
RUN apt-get update -y \
    && apt-get install -y tini \
    && pip install --upgrade pip

# Install torch and sentence-transformer
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \ 
    && pip install transformers tqdm numpy scikit-learn==1.3.0 scipy nltk sentencepiece \
    && pip install --no-deps sentence-transformers \
    && pip install -r requirements.txt

# Use tini to manage zombie processes and signal forwarding
ENTRYPOINT ["/usr/bin/tini", "--"]

# Pass the startup script as arguments to Tini
EXPOSE 8080
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT}