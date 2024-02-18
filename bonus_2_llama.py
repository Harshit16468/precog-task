import json
from llamaapi import LlamaAPI
from datasets import load_dataset
# Initialize the SDK
from tqdm import tqdm

llama = LlamaAPI("LL-TjBIYKOUDEobSmG8BgMUXNWMbDnGdSPLMvKWRPqRSrjou6cWD4PboILnJJaOjJII")
word1="old"
word2="old"
# Build the API request
api_request_json = {
    "messages": [
        {"role": "user", "content": f"Are {word1} and {word2} similar? If it is similar, give Yes, then No" },
    ],
}
print("starting to load dataset")
test_data=load_dataset("PiC/phrase_similarity", split="test")
print("dataset loaded")
results_list = []
ground_truth=[]
for i in tqdm(range(100)):
    word1=test_data[i]["phrase1"]
    word2=test_data[i]["phrase2"]
    ground_truth.append(test_data[i]["label"])
    response = llama.run(api_request_json)
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
test_accuracy = accuracy_score(ground_truth, results_list)
print("test accuracy with threshold: ", test_accuracy*100)