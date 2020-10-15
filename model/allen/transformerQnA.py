import argparse

def transformer_qna(context,question):
  from allennlp.predictors.predictor import Predictor
  import allennlp_models.rc
  predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/transformer-qa-2020-05-26.tar.gz")
  answer=predictor.predict(passage=context,question=question)["best_span_str"]
  return answer
  
parser = argparse.ArgumentParser(description="Transformer based QnA model")
parser.add_argument("question",metavar="question",type=str,nargs=1,help="Enter your question")
all_args=parser.parse_args()
question=all_args.question[0]
with open("../data/pes-corpus.txt") as infile:
    context = infile.read().strip()
t_answer=transformer_qna(context,question)
print(t_answer,"\n")
