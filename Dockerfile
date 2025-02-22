FROM tensorflow/serving:latest

COPY ./model_serving /models/garbage_classifier
ENV MODEL_NAME=garbage_classifier
EXPOSE 8501