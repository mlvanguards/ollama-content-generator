import json

import requests


class OLLAMA:
    def __init__(
        self, model_name, api_endpoint="http://localhost:11434/api/chat", **kwargs
    ):
        self.model_name = model_name
        self.api_endpoint = api_endpoint
        self.session = requests.Session()
        self.kwargs = {"temperature": 0.7, "n": 1, **kwargs}

        print(
            f"Initialized OLLAMA with model_name: {model_name}, "
            f"api_endpoint: {api_endpoint}, kwargs: {self.kwargs}"
        )

    def predict(self, question, **kwargs):
        output = ""
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": question}],
            **self.kwargs,
            **kwargs,
        }

        with self.session.post(self.api_endpoint, json=payload, stream=True) as r:
            if r.status_code == 200:
                for line in r.iter_lines():
                    if line:
                        j = json.loads(line.decode("utf-8"))
                        output += j.get("message", {}).get("content", "")
                        if j.get("done", True):
                            break
            else:
                print(f"Error: Received status code {r.status_code}")
                print(f"Response: {r.text}")

        return output.strip()

    def __call__(self, question, **kwargs):
        return self.predict(question, **kwargs)


if __name__ == "__main__":
    llama3 = OLLAMA("llama3.2:1b", temperature=0.5)
    response = llama3("What is you cappability?")
    print("llama3 Response: ", response)
