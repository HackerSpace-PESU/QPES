import argparse
import warnings
warnings.filterwarnings("ignore")

def BERT(context,question):
  import transformers
  from transformers import AutoTokenizer, TFAutoModelForQuestionAnswering
  import tensorflow as tf
  import numpy
  transformers.logging.set_verbosity_error()
  tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
  model = TFAutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
  inputs = tokenizer.encode_plus(question,context, add_special_tokens=True, return_tensors="tf")
  input_ids = inputs["input_ids"].numpy()[0] #The .numpy() method explicitly converts a Tensor to a numpy array
  text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
  answer_start_scores, answer_end_scores = model(inputs)
  answer_start = tf.argmax(answer_start_scores, axis=1).numpy()[0]  # Get the most likely beginning of answer with the argmax of the score
  answer_end = (tf.argmax(answer_end_scores, axis=1)+1).numpy()[0]  # Get the most likely end of answer with the argmax of the score,+1 cause in the list indexing the upper bound isn't included
  answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
  return answer
  
parser = argparse.ArgumentParser(description="BERT based QnA model")
parser.add_argument("question",metavar="question",type=str,nargs=1,help="Enter your question")
all_args=parser.parse_args()
question=all_args.question[0]
with open("../data/pes-corpus.txt") as infile:
    context = infile.read().strip()
context=context[:512]
bert_answer=BERT(context,question)
print(bert_answer)
