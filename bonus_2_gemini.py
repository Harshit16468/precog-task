import google.generativeai as genai
from IPython.display import display, Markdown
import json
import time
from sklearn.metrics import accuracy_score
from datasets import load_dataset
from tqdm import tqdm
print("starting to load dataset")
test_data=load_dataset("PiC/phrase_similarity", split="test")
print("dataset loaded")
genai.configure(api_key='AIzaSyB42LYqlVwiGxiHO8Rh8u1tElxNkBX82-U')

# Define the model and words.
model = genai.GenerativeModel('gemini-pro')
results_list = []
ground_truth=[]
for i in tqdm(range(100)):
    word1=test_data[i]["phrase1"]
    word2=test_data[i]["phrase2"]
    ground_truth.append(test_data[i]["label"])
    response = model.generate_content(f"are {word1} and  {word2} similar? if it similar give Yes then No")
    try:
        text = response.text
        if "Yes" in text:
            results_list.append(1)
        elif "No" in text:
            results_list.append(0)
        else:
            results_list.append(0) 

    except Exception as e:
        print(f"Error: {e}")
        results_list.append(0) 
    time.sleep(2)
test_accuracy = accuracy_score(ground_truth, results_list)
print("test accuracy with threshold: ", test_accuracy*100)
