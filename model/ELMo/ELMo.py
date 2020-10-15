

import argparse
import warnings
warnings.filterwarnings("ignore")

def elmo_embed(sentences):
  import tensorflow as tf
  import tensorflow_hub as hub
  tf.logging.set_verbosity(tf.logging.ERROR)
  embed = hub.Module('https://tfhub.dev/google/elmo/3',trainable=False)
  embeddings = embed(sentences,signature="default",as_dict=True)["default"]
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    x = sess.run(embeddings)
  return x

def ELMo(context,question):
  from nltk.tokenize import sent_tokenize
  import scipy.spatial.distance as spd
  questions=[]
  similarity=[]
  questions.append(question)
  context_sentences=sent_tokenize(context)
  context_embeddings=elmo_embed(context_sentences)
  question_embeddings=elmo_embed(questions)
  for ce in context_embeddings:
    similarity.append(1-spd.cosine(ce, question_embeddings))
  similarity_sent=list(zip(similarity,context_sentences))
  similarity_sent = sorted(similarity_sent,key=lambda x: x[0],reverse=True)
  answer=similarity_sent[0][1]
  return answer

parser = argparse.ArgumentParser(description="Simple ELMo based QnA model")
parser.add_argument("question",metavar="question",type=str,nargs=1,help="Enter your question")
all_args=parser.parse_args()
question=all_args.question[0]
with open("../data/pes-corpus.txt") as infile:
    context = infile.read().strip()
elmo_answer=ELMo(context,question)
print(elmo_answer)


