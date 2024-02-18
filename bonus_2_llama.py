import json
from llamaapi import LlamaAPI

# Initialize the SDK
llama = LlamaAPI("LL-TjBIYKOUDEobSmG8BgMUXNWMbDnGdSPLMvKWRPqRSrjou6cWD4PboILnJJaOjJII")
word1="old"
word2="old"
# Build the API request
api_request_json = {
    "messages": [
        {"role": "user", "content": f"Are {word1} and {word2} similar? If it is similar, give Yes, then No" },
    ],
}

response = llama.run(api_request_json)
try:
    text = response.text
    if "Yes" in text:
        print("Yes")
    elif "No" in text:
        print("No")
    else:
        print("No")
except Exception as e:
    print(f"Error: {e}")