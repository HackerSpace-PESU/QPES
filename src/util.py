import re
import numpy as np
from nltk import pos_tag
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk.util import ngrams


stemmer = SnowballStemmer("english")
lemmatizer = WordNetLemmatizer()
stopw = stopwords.words("english")


def stemSentence(sentence):
    words = word_tokenize(sentence)
    return " ".join([stemmer.stem(word) for word in words])


def cleanSentence(sentence, remove_stopwords=True):
    sentence = sentence.lower().strip()
    sentence = re.sub(r'[^a-z0-9\s]', '', sentence)
    if remove_stopwords:
        sentence = " ".join(
            [word for word in word_tokenize(sentence) if word not in stopw])
    return sentence


def getCleanedSentences(sentences, remove_stopwords=True):
    cleaned_sentences = []
    for i in sentences:
        cleaned = cleanSentence(i, remove_stopwords)
        stemmed = stemSentence(cleaned)
        cleaned_sentences.append(stemmed)
    return cleaned_sentences


def cosine_similarity(A, B):
    return np.dot(A, B)/(np.linalg.norm(A) * np.linalg.norm(B))


def getDoc2VecFromWord2Vec(sentence, model):
    pass


def createDocumentEmbeddingMatrix(sentences, model, mode="word2vec"):
    if mode == "doc2vec":
        result = [model.infer_vector(word_tokenize(i)) for i in sentences]
    elif mode == "word2vec":
        result = [getDoc2VecFromWord2Vec(i, model) for i in sentences]
    else:
        result = [model.transform(i) for i in sentences]
    return result


def getMatchingSentencesDescriptive(sentences, document_embedding, question_embedding, num_sentences):
    sim = [(cosine_similarity(document_embedding[i], question_embedding),
            sentences[i], i) for i in range(len(sentences))]
    sim.sort(reverse=True, key=lambda x: x[0])
    matching_sentences = sim[-num_sentences:]
    #matching_sentences.sort(key=lambda x: x[2])
    return ". ".join([i[1] for i in matching_sentences])


def getQuestionType(question):
    questionTypes = ['WP', 'WDT', 'WP$', 'WRB']
    questionPOS = pos_tag(word_tokenize(question.lower()))
    questionTags = [i for i in questionPOS if i[1] in questionTypes]
    return (len(questionTags) == 1 and questionTags[0][0] != 'what')


def n_gram_similarity(question, n):
    q = list(ngrams(word_tokenize(question.lower()), 1))
    a = 0
    b = 0
    c = 0
    t = []
    for i in q:
        if i in list(ngrams(word_tokenize(n[0].lower()), 1)):
            a = a+1
    for i in q:
        if i in list(ngrams(word_tokenize(n[1].lower()), 1)):
            b = b+1
    for i in q:
        if i in list(ngrams(word_tokenize(n[2].lower()), 1)):
            c = c+1
    d = max(a, b, c)
    if a == d:
        t.append(n[0])
    if b == d:
        t.append(n[1])
    if c == d:
        t.append(n[2])
    print()
    #print("Selected Sentence:",t[0])
    return t
