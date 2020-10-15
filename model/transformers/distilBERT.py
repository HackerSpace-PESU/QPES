
import argparse
import warnings
warnings.filterwarnings("ignore")


def distilBERT(context,question):
  import transformers
  from transformers import AutoTokenizer, TFAutoModelForQuestionAnswering
  import tensorflow as tf
  import numpy
  transformers.logging.set_verbosity(50)
  tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
  model = TFAutoModelForQuestionAnswering.from_pretrained('distilbert-base-uncased', return_dict=True)
  input_dict = tokenizer.encode_plus(question,context, add_special_tokens=True, return_tensors="tf")
  outputs = model(input_dict)
  start_logits = outputs.start_logits
  end_logits = outputs.end_logits
  all_tokens = tokenizer.convert_ids_to_tokens(input_dict["input_ids"].numpy()[0])
  answer = ' '.join(all_tokens[tf.argmax(start_logits, 1)[0] : tf.argmax(end_logits, 1)[0]+1])
  return answer

  
parser = argparse.ArgumentParser(description="distilBERT based QnA model")
parser.add_argument("question",metavar="question",type=str,nargs=1,help="Enter your question")
all_args=parser.parse_args()
question=all_args.question[0]
with open("../data/pes-corpus.txt") as infile:
    context = infile.read().strip()
context=context[:512]
bert_answer=distilBERT(context,question)
print(bert_answer)
