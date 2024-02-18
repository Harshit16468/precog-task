from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from datasets import load_dataset
word2vec_model = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)
def phrasetovector(phrase, embedding_dim=300):

    word_embeddings = []
    for word in phrase.split():
        if word in word2vec_model:
            try:
                word_embeddings.append(word2vec_model[word])
            except KeyError:
                word_embeddings.append(np.zeros(embedding_dim))
        else:
            word_embeddings.append(np.zeros(embedding_dim))

    if not word_embeddings:
        return np.zeros(embedding_dim)

    phrase_embedding = np.mean(word_embeddings, axis=0)
    return phrase_embedding
print("starting to load dataset")
dataset = load_dataset("PiC/phrase_similarity", split="train")
val_data=load_dataset("PiC/phrase_similarity", split="validation")
test_data=load_dataset("PiC/phrase_similarity", split="test")
print("dataset loaded")

model = LogisticRegression()

train_predictions = []
train_actual = []

for i in tqdm(range(len(dataset))):
    cosine_sim = cosine_similarity([phrasetovector(dataset[i]["phrase1"])], [phrasetovector(dataset[i]["phrase2"])])[0, 0]
    train_predictions.append(cosine_sim)
    train_actual.append(dataset[i]["label"])

model.fit(np.array(train_predictions).reshape(-1, 1), np.array(train_actual))

val_predictions = []
val_actual = []

for i in tqdm(range(len(val_data))):
    cosine_sim = cosine_similarity([phrasetovector(val_data[i]["phrase1"])], [phrasetovector(val_data[i]["phrase2"])])[0, 0]
    val_predictions.append(cosine_sim)
    val_actual.append(val_data[i]["label"])
val_probabilities = model.predict_proba(np.array(val_predictions).reshape(-1, 1))[:, 1]
threshold_values = np.arange(0, 1.05, 0.05)
best_threshold = 0
best_accuracy = 0
flag=0
for threshold in threshold_values:
    val_labels = (val_probabilities > threshold).astype(int)
    accuracy = accuracy_score(val_actual, val_labels)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_threshold = threshold

for threshold in threshold_values:
    
    val_labels1=(val_probabilities < threshold).astype(int)
    accuracy1=accuracy_score(val_actual, val_labels1)
    if accuracy1 > best_accuracy:
        best_accuracy = accuracy1
        best_threshold = threshold
        flag=1
for threshold in threshold_values:
    val_labels1=(val_probabilities == threshold).astype(int)
    accuracy1=accuracy_score(val_actual, val_labels1)
    if accuracy1 > best_accuracy:
        best_accuracy = accuracy1
        best_threshold = threshold
        flag=2
if flag==1:
    test_result=[]
    actual_test=[]
    for i in tqdm(range(len(test_data))):
        cosine_sim=cosine_similarity([phrasetovector(test_data[i]["phrase1"])], [phrasetovector(test_data[i]["phrase2"])])[0, 0]
        test_result.append(cosine_sim)
        actual_test.append(test_data[i]["label"])
    test_probabilities=model.predict(np.array(test_result).reshape(-1, 1))
    test_labels1=(test_probabilities < best_threshold).astype(int)
    test_accuracy = accuracy_score(actual_test, test_labels1)
    print("test accuracy with threshold: ", test_accuracy*100)
elif flag==2:
    test_result=[]
    actual_test=[]
    for i in tqdm(range(len(test_data))):
        cosine_sim=cosine_similarity([phrasetovector(test_data[i]["phrase1"])], [phrasetovector(test_data[i]["phrase2"])])[0, 0]
        test_result.append(cosine_sim)
        actual_test.append(test_data[i]["label"])
    test_probabilities=model.predict(np.array(test_result).reshape(-1, 1))
    test_labels1=(test_probabilities == best_threshold).astype(int)
    test_accuracy = accuracy_score(actual_test, test_labels1)
    print("test accuracy with threshold: ", test_accuracy*100)
else:
    test_result=[]
    actual_test=[]
    for i in tqdm(range(len(test_data))):
        cosine_sim=cosine_similarity([phrasetovector(test_data[i]["phrase1"])], [phrasetovector(test_data[i]["phrase2"])])[0, 0]
        test_result.append(cosine_sim)
        actual_test.append(test_data[i]["label"])
    test_probabilities=model.predict(np.array(test_result).reshape(-1, 1))
    test_labels1=(test_probabilities > best_threshold).astype(int)
    test_accuracy = accuracy_score(actual_test, test_labels1)
    print("test accuracy with threshold: ", test_accuracy*100)

