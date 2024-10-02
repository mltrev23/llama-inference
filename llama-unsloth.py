from unsloth import FastLanguageModel
import torch
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-Instruct-bnb-4bit", # Choose ANY! eg teknium/OpenHermes-2.5-Mistral-7B
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

text = """Sylliba is a revolutionary translation module designed to bridge the gap in communication across diverse languages. With the capability to translate many languages, Sylliba supports both audio and text for inputs and outputs, making it a versatile tool for global interactions.
As our first step into the Bittensor ecosystem, Sylliba connects to the network we are building, providing AI tooling and linking various blockchain networks together. Our mission is to create a seamless and intuitive translation experience that leverages advanced AI to foster better understanding and collaboration across different languages and cultures.
Explore Sylliba and experience the future of translation here."""
source_language = "English"
target_language = "French"

topic = "Last day on Earth"

messages = [
    {
        "role": "system",
        "content": f"""
        You are an expert story teller.
        You can write short stories that capture the imagination, 
        end readers on an adventure and complete an alegorical thought all within 100~200 words. 
        Please write a short story about {topic} in {source_language}. 
        Keep the story short but be sure to use an alegory and complete the idea.
        """
    }
]

FastLanguageModel.for_inference(model) # Enable native 2x faster inference
inputs = tokenizer([messages[0]['content']], return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 5000, use_cache = True)
result = tokenizer.batch_decode(outputs, skip_special_tokens = True)

print(result)
