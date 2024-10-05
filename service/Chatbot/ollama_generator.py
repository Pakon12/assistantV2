import json
import requests
import sys
from colorama import Fore, Style, init


# Initialize colorama
init(autoreset=True)

class OllamaChat:
    def __init__(self, model="llama3.2"):
        self.model = model
        # Combine system message with user and assistant messages
        self.system_message = self.read_file("service/Chatbot/system_message.json")
        
        if not self.system_message:
            print(Fore.RED + "ERROR: Failed to load system message.")
            return {"role": "assistant", "content": "System message could not be loaded."}
        else:
             print(Fore.GREEN + "SUCCESS: Succesfully to load system message.")

    def read_file(self, file_path):
        """Read a JSON file and return its contents."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            print(Fore.YELLOW + "WARNING: Could not find file at " + file_path)
            return {}

    def chat(self, messages):
        """Send messages to the Ollama server and get the response."""
        
        messages = [self.system_message] + messages
        
        try:
            r = requests.post(
                "http://localhost:11434/api/chat",
                json={"model": self.model, "messages": messages, "stream": True},
                stream=True
            )
            r.raise_for_status()
        except requests.RequestException as e:
            print(Fore.RED + f"ERROR: Request failed - {e}")
            return {"role": "assistant", "content": "Request failed."}

        output = ""

        for line in r.iter_lines():
            if line:
                try:
                    body = json.loads(line)
                except json.JSONDecodeError:
                    print(Fore.RED + "ERROR: Failed to decode JSON from response.")
                    continue
                
                if "error" in body:
                    print(Fore.RED + f"ERROR: {body['error']}")
                    return {"role": "assistant", "content": "An error occurred."}
                
                if body.get("done") is False:
                    message = body.get("message", {})
                    content = message.get("content", "")
                    output += content
                    # print(Fore.GREEN + content, end="", flush=True)
                    

                if body.get("done", False):
                    message["content"] = output
                    return message

# if __name__ == "__main__":
#     chat_instance = OllamaChat()
#     user_messages = [{"role": "user", "content": "Hello, how are you?"}]
#     response = chat_instance.chat(user_messages)
#     print("\nFinal Response:", response['content'])
