import util
import embedding
from nltk.tokenize import word_tokenize, sent_tokenize

with open("pescorpus.txt") as infile:
    corpus = infile.read().strip()

d2vmodel = embedding.trainDoc2VecModel(corpus)
#w2vmodel = embedding.trainWord2VecModel(corpus)
#tfidfmodel = embedding.trainTfIdfModel(corpus)

all_sentences = sent_tokenize(corpus)
#print(len(all_sentences))
embedding_matrix = util.createDocumentEmbeddingMatrix(all_sentences, d2vmodel, mode="doc2vec")

question = "What is PES?"

question_embedding = d2vmodel.infer_vector(word_tokenize(question))
answer = util.getMatchingSentencesDescriptive(all_sentences, embedding_matrix, question_embedding, 5)
print(answer)

