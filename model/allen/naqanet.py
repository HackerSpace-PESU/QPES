import argparse

def NAQANet(context,question):
  from allennlp.predictors.predictor import Predictor
  import allennlp_models.rc
  predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/naqanet-2020.02.19.tar.gz")
  answer=predictor.predict(passage=context,question=question)["answer"]["value"]
  return answer
 
parser = argparse.ArgumentParser(description="Dropnet based QnA model")
parser.add_argument("question",metavar="question",type=str,nargs=1,help="Enter your question")
all_args=parser.parse_args()
question=all_args.question[0]
with open("../data/pes-corpus.txt") as infile:
    context = infile.read().strip()
na_answer=NAQANet(context,question)
print(na_answer,"\n")
