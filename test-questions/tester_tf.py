import sys
import nltk
import transformers
from transformers import DistilBertTokenizer, TFDistilBertForQuestionAnswering
from transformers import AutoTokenizer, TFAutoModelForQuestionAnswering
import tensorflow as tf
import numpy
import warnings
warnings.filterwarnings("ignore")


def BERT(context, question):
    tokenizer = AutoTokenizer.from_pretrained(
        "bert-large-uncased-whole-word-masking-finetuned-squad")
    model = TFAutoModelForQuestionAnswering.from_pretrained(
        "bert-large-uncased-whole-word-masking-finetuned-squad")
    inputs = tokenizer.encode_plus(
        question, context, add_special_tokens=True, return_tensors="tf")
    # The .numpy() method explicitly converts a Tensor to a numpy array
    input_ids = inputs["input_ids"].numpy()[0]
    #text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    answer_start_scores, answer_end_scores = model(inputs)
    # Get the most likely beginning of answer with the argmax of the score
    answer_start = tf.argmax(answer_start_scores, axis=1).numpy()[0]
    # Get the most likely end of answer with the argmax of the score,+1 cause in the list indexing the upper bound isn't included
    answer_end = (tf.argmax(answer_end_scores, axis=1)+1).numpy()[0]
    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
    return answer


def distilBERT(context, question):
    #tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    tokenizer = AutoTokenizer.from_pretrained(
        "bert-large-uncased-whole-word-masking-finetuned-squad")
    #model = TFDistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased', return_dict=True)
    model = TFAutoModelForQuestionAnswering.from_pretrained(
        "distilbert-base-uncased")
    inputs = tokenizer.encode_plus(
        question, context, add_special_tokens=True, return_tensors="tf")
    # The .numpy() method explicitly converts a Tensor to a numpy array
    input_ids = inputs["input_ids"].numpy()[0]
    #text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    answer_start_scores, answer_end_scores = model(inputs)
    # Get the most likely beginning of answer with the argmax of the score
    answer_start = tf.argmax(answer_start_scores, axis=1).numpy()[0]
    # Get the most likely end of answer with the argmax of the score,+1 cause in the list indexing the upper bound isn't included
    answer_end = (tf.argmax(answer_end_scores, axis=1)+1).numpy()[0]
    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
    return answer


def Roberta(context, question):
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    model = TFAutoModelForQuestionAnswering.from_pretrained("roberta-base")
    inputs = tokenizer.encode_plus(
        question, context, add_special_tokens=True, return_tensors="tf")
    # The .numpy() method explicitly converts a Tensor to a numpy array
    input_ids = inputs["input_ids"].numpy()[0]
    #text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    answer_start_scores, answer_end_scores = model(inputs)
    # Get the most likely beginning of answer with the argmax of the score
    answer_start = tf.argmax(answer_start_scores, axis=1).numpy()[0]
    # Get the most likely end of answer with the argmax of the score,+1 cause in the list indexing the upper bound isn't included
    answer_end = (tf.argmax(answer_end_scores, axis=1)+1).numpy()[0]
    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
    return answer


def camemBert(context, question):
    tokenizer = AutoTokenizer.from_pretrained('camembert-base')
    model = TFAutoModelForQuestionAnswering.from_pretrained("camembert-base")
    inputs = tokenizer.encode_plus(
        question, context, add_special_tokens=True, return_tensors="tf")
    # The .numpy() method explicitly converts a Tensor to a numpy array
    input_ids = inputs["input_ids"].numpy()[0]
    #text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    answer_start_scores, answer_end_scores = model(inputs)
    # Get the most likely beginning of answer with the argmax of the score
    answer_start = tf.argmax(answer_start_scores, axis=1).numpy()[0]
    # Get the most likely end of answer with the argmax of the score,+1 cause in the list indexing the upper bound isn't included
    answer_end = (tf.argmax(answer_end_scores, axis=1)+1).numpy()[0]
    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
    return answer


def flauBert(context, question):
    tokenizer = AutoTokenizer.from_pretrained('flaubert/flaubert_base_uncased')
    model = TFAutoModelForQuestionAnswering.from_pretrained(
        "flaubert/flaubert_base_uncased")
    inputs = tokenizer.encode_plus(
        question, context, add_special_tokens=True, return_tensors="tf")
    # The .numpy() method explicitly converts a Tensor to a numpy array
    input_ids = inputs["input_ids"].numpy()[0]
    #text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    answer_start_scores, answer_end_scores = model(inputs)
    # Get the most likely beginning of answer with the argmax of the score
    answer_start = tf.argmax(answer_start_scores, axis=1).numpy()[0]
    # Get the most likely end of answer with the argmax of the score,+1 cause in the list indexing the upper bound isn't included
    answer_end = (tf.argmax(answer_end_scores, axis=1)+1).numpy()[0]
    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
    return answer


def bart(context, question):
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
    model = TFAutoModelForQuestionAnswering.from_pretrained(
        "facebook/bart-base")
    inputs = tokenizer.encode_plus(
        question, context, add_special_tokens=True, return_tensors="tf")
    # The .numpy() method explicitly converts a Tensor to a numpy array
    input_ids = inputs["input_ids"].numpy()[0]
    #text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    answer_start_scores, answer_end_scores = model(inputs)
    # Get the most likely beginning of answer with the argmax of the score
    answer_start = tf.argmax(answer_start_scores, axis=1).numpy()[0]
    # Get the most likely end of answer with the argmax of the score,+1 cause in the list indexing the upper bound isn't included
    answer_end = (tf.argmax(answer_end_scores, axis=1)+1).numpy()[0]
    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
    return answer


def long_former(context, question):
    tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
    model = TFAutoModelForQuestionAnswering.from_pretrained(
        "allenai/longformer-base-4096")
    inputs = tokenizer.encode_plus(
        question, context, add_special_tokens=True, return_tensors="tf")
    # The .numpy() method explicitly converts a Tensor to a numpy array
    input_ids = inputs["input_ids"].numpy()[0]
    #text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    answer_start_scores, answer_end_scores = model(inputs)
    # Get the most likely beginning of answer with the argmax of the score
    answer_start = tf.argmax(answer_start_scores, axis=1).numpy()[0]
    # Get the most likely end of answer with the argmax of the score,+1 cause in the list indexing the upper bound isn't included
    answer_end = (tf.argmax(answer_end_scores, axis=1)+1).numpy()[0]
    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
    return answer


def xlm_roberta(context, question):
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    model = TFAutoModelForQuestionAnswering.from_pretrained("xlm-roberta-base")
    inputs = tokenizer.encode_plus(
        question, context, add_special_tokens=True, return_tensors="tf")
    # The .numpy() method explicitly converts a Tensor to a numpy array
    input_ids = inputs["input_ids"].numpy()[0]
    #text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    answer_start_scores, answer_end_scores = model(inputs)
    # Get the most likely beginning of answer with the argmax of the score
    answer_start = tf.argmax(answer_start_scores, axis=1).numpy()[0]
    # Get the most likely end of answer with the argmax of the score,+1 cause in the list indexing the upper bound isn't included
    answer_end = (tf.argmax(answer_end_scores, axis=1)+1).numpy()[0]
    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
    return answer


def reformer(context, question):
    tokenizer = AutoTokenizer.from_pretrained("reformer-enwik8")
    model = TFAutoModelForQuestionAnswering.from_pretrained("reformer-enwik8")
    inputs = tokenizer.encode_plus(
        question, context, add_special_tokens=True, return_tensors="tf")
    # The .numpy() method explicitly converts a Tensor to a numpy array
    input_ids = inputs["input_ids"].numpy()[0]
    #text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    answer_start_scores, answer_end_scores = model(inputs)
    # Get the most likely beginning of answer with the argmax of the score
    answer_start = tf.argmax(answer_start_scores, axis=1).numpy()[0]
    # Get the most likely end of answer with the argmax of the score,+1 cause in the list indexing the upper bound isn't included
    answer_end = (tf.argmax(answer_end_scores, axis=1)+1).numpy()[0]
    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
    return answer


def funnel(context, question):
    tokenizer = AutoTokenizer.from_pretrained(
        "funnel-transformer/intermediate-base")
    model = TFAutoModelForQuestionAnswering.from_pretrained(
        "funnel-transformer/intermediate-base")
    inputs = tokenizer.encode_plus(
        question, context, add_special_tokens=True, return_tensors="tf")
    # The .numpy() method explicitly converts a Tensor to a numpy array
    input_ids = inputs["input_ids"].numpy()[0]
    #text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    answer_start_scores, answer_end_scores = model(inputs)
    # Get the most likely beginning of answer with the argmax of the score
    answer_start = tf.argmax(answer_start_scores, axis=1).numpy()[0]
    # Get the most likely end of answer with the argmax of the score,+1 cause in the list indexing the upper bound isn't included
    answer_end = (tf.argmax(answer_end_scores, axis=1)+1).numpy()[0]
    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
    return answer


with open("../data/pes-corpus.txt") as datafile:
    context = datafile.read().strip()

models = {'BERT': BERT, 'Roberta': Roberta, 'camemBert': camemBert, 'distilBERT': distilBERT, 'flauBert': flauBert,
          'funnel': funnel, 'bart': bart, 'long_former': long_former, 'reformer': reformer, 'xlm_roberta': xlm_roberta}


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
