# Word2Vec written in pure Numpy (demo)
# SKip-Gram Module
# author: aaaddress1@chroot.org
import re, random
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
vocab = {}
for line in corpus:
    for word in line:
        if not word in vocab:
            vocab[word] = make_small_ndarray(50)
            print(f'[+] append word - {word}')

# train.


for epoch in range(1, 11):
    lr = 0.025 * (1 / epoch)
    for line in tqdm(corpus):
        for pos in range(1, len(line) - 1):
            current = vocab[ line[pos] ]
            context = vocab[line[pos - 1]] + vocab[line[pos + 1]]

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

    # display the fucking power.
    predict_queen = vocab['man'] - vocab['woman'] + vocab['king']
    print('[+] man:woman = king:queen ... success? %.3f' % cosine_similarity(predict_queen, vocab['queen']))
    print('similar(king, queen) = %.3f' % cosine_similarity(vocab['king'], vocab['queen']))
    print('similar(man, woman) = %.3f' % cosine_similarity(vocab['man'], vocab['woman']))
    print('similar(man, dog) = %.3f' % cosine_similarity(vocab['man'], vocab['dog']))
