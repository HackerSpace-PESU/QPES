import allennlp_models.rc
from allennlp.predictors.predictor import Predictor
import nltk
import sys
import os
import warnings
warnings.filterwarnings("ignore")


def elmo_Bidaf(context, question):
    predictor = Predictor.from_path(
        "https://storage.googleapis.com/allennlp-public-models/bidaf-elmo-model-2020.03.19.tar.gz")
    answer = predictor.predict(passage=context, question=question)[
        'best_span_str']
    return answer


def Bidaf(context, question):
    predictor = Predictor.from_path(
        "https://storage.googleapis.com/allennlp-public-models/bidaf-model-2020.03.19.tar.gz")
    answer = predictor.predict(passage=context, question=question)[
        "best_span_str"]
    return answer


def transformer_qna(context, question):
    predictor = Predictor.from_path(
        "https://storage.googleapis.com/allennlp-public-models/transformer-qa-2020-05-26.tar.gz")
    answer = predictor.predict(passage=context, question=question)[
        "best_span_str"]
    return answer


def NAQANet(context, question):
    predictor = Predictor.from_path(
        "https://storage.googleapis.com/allennlp-public-models/naqanet-2020.02.19.tar.gz")
    answer = predictor.predict(passage=context, question=question)[
        "answer"]["value"]
    return answer


with open("../data/pes-corpus.txt") as datafile:
    context = datafile.read().strip()

models = {'elmo_Bidaf': elmo_Bidaf, 'Bidaf': Bidaf,
          'transformer_qna': transformer_qna, 'NAQANet': NAQANet}

question, model_type = sys.argv[1], sys.argv[2]
try:
    model_function = models[model_type]
    print(model_function(context, question))
except:
    print(f"ERROR")

"""with open("questions.txt") as Q:
    questions = Q.read().strip().split('\n')
for question_type in questions:
    print(question_type)
    for model_type in models.keys():
        try:
            print(models[model_type](context, question_type))
        except:
            print(f"Error in model: {model_type}")
    break"""
