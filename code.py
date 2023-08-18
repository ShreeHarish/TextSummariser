from numpy import array
from nltk import sent_tokenize, word_tokenize, pos_tag
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
import re
from nltk.probability import FreqDist
from numpy import divide, zeros, array, diag
from numpy.linalg import svd
import math

class PreProcessingModule:

def __init__(self, filename):
    # Open the file and read the input as string
    file = open(filename)
    self.Paragraph = file.read()
    file.close()
    # Creating the sentences array from the given paragraph
    self.Sentences = array(sent_tokenize(self.Paragraph))
    # Removing punctuation from tokenized sentences and lowercase them
    self.Lower_Sentences = self.__punctuation_removal(self.Sentences)
    # Creating the words array from the given paragraph
    self.Words = array(word_tokenize(self.Paragraph))
    # Removing punctuation from tokenized words and lowercase them
    lower_words = []
    for sentence in self.Lower_Sentences:
        words = word_tokenize(sentence)
        words = self.__punctuation_removal(words)
        lower_words.extend(words)
    self.Lower_Words = lower_words
    # Creating the words in sentences array because words may change
    self.Words_in_Sentences = {}
    # Creating the words - stem words mapping
    self.Stemmed_Words = {}
    # Creating the postag() to parts-of-speech tagging
    self.__POSDict = defaultdict(lambda: wordnet.NOUN)
    self.__POSDict['J'] = wordnet.ADJ
    self.__POSDict['V'] = wordnet.VERB
    self.__POSDict['R'] = wordnet.ADV

    # Creating the word to postag mapping
    self.__POS = defaultdict(lambda: wordnet.NOUN)
    # Finding words in final processed sentences
    for sentence in self.Lower_Sentences:
        words = word_tokenize(sentence)
        words = self.__punctuation_removal(words)
    self.Words_in_Sentences[sentence] = words

@staticmethod
def __punctuation_removal(tokens):
    # RE for removing punctuation
    punctuation = re.compile(r'[.?`\'$"!%,:;()|0-9]')
    hyphens = re.compile(r'^-')
    s = re.compile(r'^s$')
    # Removing punctuation from tokens and lowercase them
    post_punctuation = []
    for token in tokens:
        token = punctuation.sub("", token.lower())
        token = hyphens.sub("", token)
        token = s.sub("", token)
        if len(token) > 0:
            post_punctuation.append(token)
    return post_punctuation

def stop_words_removal(self):
    # get stopwords from nltk library
    stop_words = set(stopwords.words('english'))
    stop_words.add('bn')
    stop_words.add('mr')
    # Removal of words from __Lower_Words
    filtered_tokens = [w for w in self.Lower_Words if w not in stop_words]
    # making filtered_tokens as default tokens
    self.Lower_Words = filtered_tokens

def pos_tagging(self):
    # Finding and storing POS of tokens
    for sentence in self.Lower_Sentences:
        words = self.Words_in_Sentences[sentence]
        pos_tags = list(pos_tag(words))
        for item in pos_tags:
            self.__POS[item[0]] = self.__POSDict[item[1][0]]

def word_stemming(self):
    # Initializing word stemmer
    word_lemmatizer = WordNetLemmatizer()
    # Initializing stem removed lowercase words
    lower_words = []
    # Finding the stem of words and storing it
    for sentence in self.Lower_Sentences:
        sentence_words = self.Words_in_Sentences[sentence]
        for token in sentence_words:
            stem = word_lemmatizer.lemmatize(token, self.__POS[token])
        self.Stemmed_Words[token] = stem
    for token in self.Lower_Words:
        stem = word_lemmatizer.lemmatize(token, self.__POS[token])
    if stem not in lower_words:
        lower_words.append(stem)
    self.Lower_Words = lower_words


class LatentSemanticAnalysisModule:
    def __init__(self, lower_sentences, lower_words, stemmed_words,
        words_in_sentences):
        # Initialization with outputs of preprocessing module
        self.__Lower_Sentences = lower_sentences
        self.__Lower_Words = lower_words
        self.__Stemmed_Words = stemmed_words
        self.__Words_in_Sentences = words_in_sentences
        # Count of Document Words
        document_count = 0
        for _, val in self.__Words_in_Sentences.items():
            document_count += len(val)
        self.Document_Count = document_count
        # Count of Sentences and words
        self.__Sentences_Count = len(self.__Lower_Sentences)
        self.__Words_Count = len(self.__Lower_Words)
        # Word to their corresponding count mapping dictionary
        count_dict = FreqDist()
        for sentence in self.__Lower_Sentences:
            words = self.__Words_in_Sentences[sentence]
            for word in words:
                count_dict[word] += 1
        self.__Count_Dict = count_dict
            # Find the number of sentences with the word mapping
        self.__No_of_Sentences_with_Word = {}
        self.Input_Matrix = [[0 for i in range(self.__Sentences_Count)] for j in range(self.__Words_Count)]
        self.U = []
        self.S = []
        self.Vt = []

    def __binary_representation(self):
        # iterate over preprocessed words
        for i in range(self.__Words_Count):
        # iterate over lowercase sentences
            for j in range(self.__Sentences_Count):
            # find the words in sentence
            sentence_words = self.__Words_in_Sentences[self.__Lower_Sentences[j]]
        # if word is present change the entry to 1
        for word in sentence_words:
            if self.__Stemmed_Words[word] == self.__Lower_Words[i]:
                self.Input_Matrix[i][j] = 1

    def __tfidf_representation(self):
        for i in range(self.__Words_Count):
        # iterate over lowercase sentences
            for j in range(self.__Sentences_Count):
            # find the words in sentence
                sentence_words = self.__Words_in_Sentences[self.__Lower_Sentences[j]]
            # if word is present increase the entry to 1
            for word in sentence_words:
                if self.__Stemmed_Words[word] == self.__Lower_Words[i]:
                    self.Input_Matrix[i][j] += 1
        # find the number of sentences with words count
        for i in range(self.__Words_Count):
            count = 0
            for j in range(self.__Sentences_Count):
                if self.Input_Matrix[i][j] != 0:
                    count += 1
        self.__No_of_Sentences_with_Word[self.__Lower_Words[i]] = count
        # term frequency(t) = number of times word t appears in a document/ total number of words in the document
        self.Input_Matrix = divide(self.Input_Matrix,
        self.Document_Count)
        for i in range(self.__Words_Count):
            for j in range(self.__Sentences_Count):
        # inverse document frequency(t) = log( total no of documents / no of documents with term t )
            idf = math.log(self.__Sentences_Count / self.__No_of_Sentences_with_Word[self.__Lower_Words[i]])
            self.Input_Matrix[i][j] = self.Input_Matrix[i][j] * idf

    def input_matrix_creation(self, method="Binary"):
        # Check which technique to use
        if method == "Binary":
            self.__binary_representation()
        else:
            self.__tfidf_representation()

    def __validate_num_of_concepts(self, num_of_concepts):
        # Find the number of linearly independent sentences
        # To calculate the rank of the input matrix
        set_of_sentences = set(frozenset(sentence.split(' ')) for
        sentence in self.__Lower_Sentences)
        input_matrix_rank = len(set_of_sentences)
        # Validating num_of_concepts by comparing it with rank of the input_matrix_rank
        if input_matrix_rank <= 1:
            print("The input matrix does not have the ranks to compute SVD")
            exit(-1)
        if num_of_concepts > input_matrix_rank - 1:
            num_of_concepts = input_matrix_rank - 1
        return num_of_concepts

    def singular_value_decomposition(self, num_of_concepts=5):
# SVD Calculation
        self.U, self.S, self.Vt = svd(self.Input_Matrix, full_matrices=False)
        num_of_concepts= self.__validate_num_of_concepts(num_of_concepts)
        # Resizing SVD matrix based on num_of_concepts
        self.Vt = self.Vt[:num_of_concepts, :]
        self.U = self.U[:, :num_of_concepts]
        self.S = self.S[0:num_of_concepts]


if __name__ == '__main__':
#PREPROCESSING MODULE
    preprocessing = PreProcessingModule('input.txt')
    # Removal of stop words
    preprocessing.stop_words_removal()
    # Tagging of Parts-of-Speech
    preprocessing.pos_tagging()
    # Stemming of words
    preprocessing.word_stemming()
    # Results of Preprocessing Module
    # Paragraph, Sentences, Words, Lower_Sentences, Lower_Words,
    Words_in_Sentences, Stemmed_Words
    #LATENT SEMANTIC ANALYSIS MODULE
    lsa = LatentSemanticAnalysisModule(preprocessing.Lower_Sentences, preprocessing.Lower_Words, preprocessing.Stemmed_Words, preprocessing.Words_in_Sentences)
    # Input Matrix Creation using Binary Representation
    lsa.input_matrix_creation(method="Term")
    # Singular value decomposition
    lsa.singular_value_decomposition()
    # Results of Latent Semantic Analysis Module
    # Input Matrix, Singular Value Decomposition ( Input_Matrix = U . S . Vt )
    print(lsa.U.shape)
    print(lsa.S.shape)
    print(lsa.Vt.shape)
