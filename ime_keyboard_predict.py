# Word2Vec to Predict Next Word (like IME)
# idea: https://jalammar.github.io/illustrated-word2vec/
# author: aaaddress1@chroot.org
import re, random, pickle
import numpy as np
from tqdm import tqdm

def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
def make_small_ndarray(dim: int) -> np.ndarray:
    rng = np.random.default_rng()
    return (rng.random(dim) - 0.5) / dim

# get corpus.
data = open('alice.txt', 'r').read()
corpus = []
for x in data.splitlines():
    line = [ word for word in x.strip().split('\x20') if len(word) > 1 ]
    line = [ re.sub(r'[^\w]', '', c).lower() for c in line ]
    if len(line) > 2:
        corpus.append(line)

# vocab.


try:
    fr = open("vocab.bin", "rb")
    vocab = pickle.load(fr)
    btrain = True
except:
    btrain = False
    vocab = {}
    for line in corpus:
        for word in line:
            if not word in vocab:
                vocab[word] = make_small_ndarray(50)
                print(f'[+] append word - {word}')


# train.
if not btrain or 'Y' == input("train more? (Y/N): "):
    for epoch in range(1, 5):
        lr = 0.25 * (1 / epoch)
        for line in tqdm(corpus):
            for pos in range(1, len(line) - 1):
                current = vocab[ line[pos + 1] ]
                context = vocab[line[pos - 1]] + vocab[line[pos]] 

                # postive sampling.
                grad = (1 - sigmoid(current.dot(current * context))) * current * lr 
                vocab[line[pos - 1]] -= (1 / 2) * grad
                vocab[line[pos + 1]] -= (1 / 2) * grad

                # negative sampling.
                for negative_word in random.choices(list(vocab), k=25):
                    current = vocab[ negative_word ]
                    grad = sigmoid(current.dot(current * context)) * current * lr 
                    vocab[line[pos - 1]] -= (1 / 2) * grad
                    vocab[line[pos + 1]] -= (1 / 2) * grad

def findSimilarWord_byVec(in_vector, top_k = 5):
    tmp_score, tmp_token = -99, None
    ret = {tk : cosine_similarity(in_vector, vocab[tk]) for tk in vocab}
    ret = sorted(ret.items(), key=lambda item: item[1], reverse=True)[:top_k]
    return ret

def predict_nextWord(sentence_of2words, top_k = 10):
    word1, word2 = sentence_of2words.split()[:2]
    rlist= findSimilarWord_byVec( (vocab[word1.lower()] + vocab[word2.lower()]) *2 , top_k )
    print( '\n'.join([  f"{elmt[0]}({elmt[1]:.2f})" for elmt in rlist]) )

pickle.dump(vocab, open("vocab.bin", "wb"))

import IPython
IPython.embed()
