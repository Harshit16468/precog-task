from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from xgboost import XGBClassifier

model1 = SentenceTransformer("all-MiniLM-L6-v2")
dataset1 = load_dataset("paws", "labeled_final")

def phrasetovector(sentence, embedding_dim=300):
    embedding1 = model1.encode(sentence)
    sentence_embedding = np.array(embedding1).reshape(1, -1)
    return sentence_embedding[0]

print("starting to load dataset")
dataset = dataset1["train"]
test_data = dataset1["test"]
print("dataset loaded")

train_predictions = []
train_actual = []

for i in tqdm(range(len(dataset))):
    cosine_sim = cosine_similarity([phrasetovector(dataset[i]["sentence1"])], [phrasetovector(dataset[i]["sentence2"])])[0, 0]
    train_predictions.append(cosine_sim)
    train_actual.append(dataset[i]["label"])

model = XGBClassifier()
model.fit(np.array(train_predictions).reshape(-1, 1), np.array(train_actual))

test_result = []
actual_test = []

for i in tqdm(range(len(test_data))):
    cosine_sim = cosine_similarity([phrasetovector(test_data[i]["sentence1"])], [phrasetovector(test_data[i]["sentence2"])])[0, 0]
    test_result.append(cosine_sim)
    actual_test.append(test_data[i]["label"])

test_probabilities = model.predict(np.array(test_result).reshape(-1, 1))
test_accuracy = accuracy_score(actual_test, test_probabilities)
print("Test accuracy: ", test_accuracy * 100)
