from transformers import AutoTokenizer
import transformers 
import torch
model = "TinyLlama/TinyLlama_v1.1"
tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

sequences = pipeline(
    '너 한국어 할 줄 알아?',
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    repetition_penalty=1.5,
    eos_token_id=tokenizer.eos_token_id,
    truncation=True,
    max_length=500,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")
