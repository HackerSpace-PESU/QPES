from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer


def trainWord2VecModel(corpus):
    all_sentences = sent_tokenize(corpus)
    training_data = [word_tokenize(sentence) for sentence in all_sentences]
    model = Word2Vec(min_count=1, window=3, size=300)
    model.build_vocab(training_data, progress_per=1000)
    model.train(training_data, total_examples=model.corpus_count, epochs=50)
    return model


def trainDoc2VecModel(corpus):
    all_sentences = sent_tokenize(corpus)
    training_data = [TaggedDocument(words=word_tokenize(sentence), tags=[
                                    str(i)]) for i, sentence in enumerate(all_sentences)]
    model = Doc2Vec(vector_size=500, window=3, epochs=50, min_count=1)
    model.build_vocab(training_data)
    model.train(training_data, total_examples=model.corpus_count, epochs=50)
    return model


def trainTfIdfModel(corpus):
    all_sentences = sent_tokenize(corpus)
    model = TfidfVectorizer()
    model.fit(all_sentences)
    return model
