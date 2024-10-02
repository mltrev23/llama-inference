# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("LoneStriker/Reflection-Llama-3.1-70B-4.0bpw-h6-exl2")
model = AutoModelForCausalLM.from_pretrained(
    "LoneStriker/Reflection-Llama-3.1-70B-4.0bpw-h6-exl2",
    quantization_config="bitsandbytes_4bit"
)

messages = 'hello what are you doing?'

# inference using tokenizer and model
inputs = tokenizer(messages, return_tensors="pt").input_ids
response = model.generate(inputs)
print(tokenizer.decode(response[0], skip_special_tokens=True))