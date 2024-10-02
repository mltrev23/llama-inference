from llama_cpp import Llama

llm = Llama.from_pretrained(
	repo_id="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
	filename="Meta-Llama-3.1-8B-Instruct-IQ2_M.gguf",
)

result = llm.create_chat_completion(
	messages = [
		{
			"role": "user",
			"content": "What is the capital of France?"
		}
	]
)

print(result['choices'][0]['message']['content'])