import numpy as np
import string
from datasets import load_dataset
import nltk
import random
import spacy
import re

nlp = spacy.load('en_core_web_sm')

class BaseModel(object):
    def __init__(self, N, window_size):
        self.N = N
        self.X_train = []
        self.y_train = []
        self.window_size = window_size
        self.alpha = 0.01
        self.words = []
        self.word_index = {}

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def initialize(self, V, data):
        self.V = V
        self.W1 = np.random.uniform(-0.8, 0.8, (self.V, self.N))
        self.W2 = np.random.uniform(-0.8, 0.8, (self.N, self.V))
        self.words = data
        for i in range(len(data)):
            self.word_index[data[i]] = i

    def preprocessing(self, corpus):
        roman_numeral_pattern = r'\bM{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})\b'
        corpus = re.sub(roman_numeral_pattern, '', corpus)
        doc = nlp(corpus)
        cleaned_tokens = []
        for sent in doc.sents:
            temp_tokens = []
            for token in sent:
                if not token.is_alpha:
                    continue
                if token.is_punct or token.like_num or not token.is_ascii:
                    continue
                temp_tokens.append(token.lower_)
            if temp_tokens:
                cleaned_tokens.append(temp_tokens)
        return cleaned_tokens

    def feed_forward(self, X):
        self.h = np.dot(self.W1.T, X).reshape(self.N, 1)
        self.u = np.dot(self.W2.T, self.h)
        self.y = self.softmax(self.u)
        return self.y

    def backpropagate(self, x, t):
        e = self.y - np.asarray(t).reshape(self.V, 1)
        dLdW2 = np.dot(self.h, e.T)
        X = np.array(x).reshape(self.V, 1)
        dLdW1 = np.dot(X, np.dot(self.W2, e).T)
        self.W2 = self.W2 - self.alpha * dLdW2
        self.W1 = self.W1 - self.alpha * dLdW1

    def compute_similarity(self, vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def rank_words(self, target_vector):
        similarities = {}
        for i in range(self.V):
            word_vector = self.W1[i]
            similarity = self.compute_similarity(target_vector, word_vector)
            similarities[i] = similarity
        ranked_words = sorted(similarities, key=similarities.get, reverse=True)
        return ranked_words

    def compute_mrr_for_window(self, target_word, context_words):
        target_index = self.word_index[target_word]
        target_vector = self.W1[target_index]

        ranked_indices = self.rank_words(target_vector)

        mrr = 0
        for context_word in context_words:
            context_index = self.word_index[context_word]
            rank = ranked_indices.index(context_index) + 1
            mrr += 1 / rank
        return mrr / len(context_words)

    def compute_mrr(self, test_data):
        total_mrr = 0
        for window in test_data:
            target_word, context_words = window[0], window[1:]
            total_mrr += self.compute_mrr_for_window(target_word, context_words)
        average_mrr = total_mrr / len(test_data)
        return average_mrr

    def evaluate_mrr(self, test_corpus):
        test_sentences = self.preprocessing(test_corpus)
        test_data = []
        for sentence in test_sentences:
            for i in range(len(sentence)):
                window = sentence[max(0, i - self.window_size):min(len(sentence), i + self.window_size + 1)]
                test_data.append(window)

        mrr = self.compute_mrr(test_data)
        print("MRR:", mrr)

class SkipGramSoftMax(BaseModel):
    def prepare_data_for_training(self, corpus):
        sentences = self.preprocessing(corpus)
        data = {}
        for sentence in sentences:
            for word in sentence:
                if word not in data:
                    data[word] = 1
                else:
                    data[word] += 1

        V = len(data)
        data = sorted(list(data.keys()))
        vocab = {data[i]: i for i in range(len(data))}

        for sentence in sentences:
            for i in range(len(sentence)):
                center_word = [0 for _ in range(V)]
                center_word[vocab[sentence[i]]] = 1
                context = [0 for _ in range(V)]
                for j in range(i - self.window_size, i + self.window_size + 1):
                    if i != j and j >= 0 and j < len(sentence):
                        context[vocab[sentence[j]]] += 1
                self.X_train.append(center_word)
                self.y_train.append(context)

        self.initialize(V, data)

    def train(self, epochs):
        for epoch in range(1, epochs + 1):
            self.loss = 0
            for j in range(len(self.X_train)):
                self.feed_forward(self.X_train[j])
                self.backpropagate(self.X_train[j], self.y_train[j])
                C = 0
                for m in range(self.V):
                    if self.y_train[j][m]:
                        self.loss += -1 * self.u[m][0]
                        C += 1
                self.loss += C * np.log(np.sum(np.exp(self.u)))
            if epoch % 1000 == 0:
                print("Epoch", epoch, "loss =", self.loss)
            self.alpha *= 1 / (1 + self.alpha * epoch)

    def predict(self, word, number_of_predictions):
        if word in self.words:
            index = self.word_index[word]
            X = [0 for _ in range(self.V)]
            X[index] = 1
            prediction = self.feed_forward(X)
            output = {}
            for i in range(self.V):
                output[prediction[i][0]] = i
            top_context_words = []
            for k in sorted(output, reverse=True):
                top_context_words.append(self.words[output[k]])
                if len(top_context_words) >= number_of_predictions:
                    break
            print("Word:", word, ", Context words:", top_context_words)
        else:
            print("Word not found in dictionary")

class CBOWSoftMax(BaseModel):
    def prepare_data_for_training(self, corpus):
        sentences = self.preprocessing(corpus)
        data = {}
        for sentence in sentences:
            for word in sentence:
                if word not in data:
                    data[word] = 1
                else:
                    data[word] += 1
        V = len(data)
        data = sorted(list(data.keys()))
        vocab = {}
        for i in range(len(data)):
            vocab[data[i]] = i

        for sentence in sentences:
            for i in range(len(sentence)):
                context = [0 for x in range(V)]
                center_word = [0 for x in range(V)]
                center_word[vocab[sentence[i]]] = 1
                for j in range(i - self.window_size, i + self.window_size + 1):
                    if i != j and j >= 0 and j < len(sentence):
                        context[vocab[sentence[j]]] += 1
                self.X_train.append(context)
                self.y_train.append(center_word)

        self.initialize(V, data)
        return self.X_train, self.y_train

    def train(self, epochs):
        for epoch in range(1, epochs + 1):
            self.loss = 0
            for j in range(len(self.X_train)):
                self.feed_forward(self.X_train[j])
                self.backpropagate(self.X_train[j], self.y_train[j])
                C = 0
                for m in range(self.V):
                    if self.y_train[j][m]:
                        self.loss += -1 * self.u[m][0]
                        C += 1
                self.loss += C * np.log(np.sum(np.exp(self.u)))
            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Loss: {self.loss}")
            self.alpha *= 1 / (1 + self.alpha * epoch)

    def predict(self, context_words, number_of_predictions):
        X = [0 for i in range(self.V)]
        for word in context_words:
            if word in self.words:
                index = self.word_index[word]
                X[index] += 1

        prediction = self.feed_forward(X)
        output = {}
        for i in range(self.V):
            output[prediction[i][0]] = i
        top_predictions = []
        for k in sorted(output, reverse=True):
            top_predictions.append(self.words[output[k]])
            if len(top_predictions) >= number_of_predictions:
                break
        print("Context words:", context_words, ", Predicted word:", top_predictions)

class SkipGramNegativeSampling(BaseModel):
    def prepare_data_for_training(self, corpus):
        sentences = self.preprocessing(corpus)
        data = {}
        for sentence in sentences:
            for word in sentence:
                if word not in data:
                    data[word] = 1
                else:
                    data[word] += 1

        V = len(data)
        data = sorted(list(data.keys()))
        vocab = {data[i]: i for i in range(len(data))}

        for sentence in sentences:
            for i in range(len(sentence)):
                center_word = [0 for _ in range(V)]
                center_word[vocab[sentence[i]]] = 1
                context = [0 for _ in range(V)]
                for j in range(i - self.window_size, i + self.window_size + 1):
                    if i != j and j >= 0 and j < len(sentence):
                        context[vocab[sentence[j]]] += 1
                self.X_train.append(center_word)
                self.y_train.append(context)

        self.initialize(V, data)

    def get_negative_samples(self, target_word_index, num_samples):
        neg_samples = []
        while len(neg_samples) < num_samples:
            neg_word_index = random.randint(0, self.V - 1)
            if neg_word_index != target_word_index:
                neg_samples.append(neg_word_index)
        return neg_samples

    def train(self, epochs, negative_samples):
        for epoch in range(1, epochs + 1):
            self.loss = 0
            for j in range(len(self.X_train)):
                center_word_vector = self.X_train[j]
                context_vector = self.y_train[j]

                self.feed_forward(center_word_vector)
                self.backpropagate(center_word_vector, context_vector)

                center_word_index = np.argmax(center_word_vector)
                neg_samples = self.get_negative_samples(center_word_index, negative_samples)

                for neg_index in neg_samples:
                    neg_context_vector = [0] * self.V
                    neg_context_vector[neg_index] = 1
                    self.feed_forward(center_word_vector)
                    self.backpropagate(center_word_vector, neg_context_vector)

                C = 0
                for m in range(self.V):
                    if context_vector[m]:
                        self.loss += -1 * self.u[m][0]
                        C += 1
                self.loss += C * np.log(np.sum(np.exp(self.u)))
            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Loss: {self.loss}")
            self.alpha *= 1 / (1 + self.alpha * epoch)

    def predict(self, word, number_of_predictions):
        if word in self.words:
            index = self.word_index[word]
            X = [0 for _ in range(self.V)]
            X[index] = 1
            prediction = self.feed_forward(X)
            output = {}
            for i in range(self.V):
                output[prediction[i][0]] = i
            top_context_words = []
            for k in sorted(output, reverse=True):
                top_context_words.append(self.words[output[k]])
                if len(top_context_words) >= number_of_predictions:
                    break
            print("Word:", word, ", Context words:", top_context_words)
        else:
            print("Word not found in dictionary")

train_corpus = "The earth revolves around the sun. The moon revolves around the earth."
test_corpus = "The sun revolves around the earth. The earth revolves around the moon."

embeddingSize = 50
windowSize = 2
epochs = 10000
negativeSamples = 5

skipGramSoftMax = SkipGramSoftMax(N=embeddingSize, window_size=windowSize)
skipGramSoftMax.prepare_data_for_training(train_corpus)
skipGramSoftMax.train(epochs=epochs)

cbowSoftMax = CBOWSoftMax(N=embeddingSize, window_size=windowSize)
cbowSoftMax.prepare_data_for_training(train_corpus)
cbowSoftMax.train(epochs=epochs)

skipGramNegativeSampling = SkipGramNegativeSampling(N=embeddingSize, window_size=windowSize)
skipGramNegativeSampling.prepare_data_for_training(train_corpus)
skipGramNegativeSampling.train(epochs=epochs, negative_samples=negativeSamples)

skipGramSoftMax.evaluate_mrr(test_corpus)
skipGramSoftMax.predict("earth", 5)
# print(skipGramSoftMax.W1[skipGramSoftMax.word_index["earth"]])

cbowSoftMax.evaluate_mrr(test_corpus)
cbowSoftMax.predict(["around", "sun"], 5)
# print(cbowSoftMax.W1[cbowSoftMax.word_index["around"]])

skipGramNegativeSampling.evaluate_mrr(test_corpus)
skipGramNegativeSampling.predict("earth", 5)
# print(skipGramNegativeSampling.W1[skipGramNegativeSampling.word_index["earth"]])