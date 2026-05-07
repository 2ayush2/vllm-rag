from openai import OpenAI

BASE_URL = "http://localhost:8000/v1"
MODEL_ID = "./models/qwen3-4b-gguf/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
SYSTEM_PROMPT = "You are a helpful AI assistant. Respond clearly and concisely."

class VLLMInference:
    def __init__(self):
        self.client = OpenAI(base_url=BASE_URL, api_key="test-key")
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    def get_response(self, user_input: str):
        self.messages.append({"role": "user", "content": user_input})
        stream = self.client.chat.completions.create(model=MODEL_ID, messages=self.messages, stream=True)
        full_response = ""

        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                full_response += content
                yield content

        self.messages.append({"role": "assistant", "content": full_response})


if __name__ == "__main__":
    bot = VLLMInference()
    print("Type 'exit' to quit")

    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        print("Assistant: ", end="", flush=True)
        try:
            for token in bot.get_response(user_input):
                print(token, end="", flush=True)
            print()
        except Exception as e:
            print("Error:", e)