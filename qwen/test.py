from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-7B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "씨발!!!!!"
print(f'prompt: {prompt}')

messages = [
    {"role": "system", "content": "당신은 뛰어난 심리상담가입니다. 사용자의 감정적인 상태가 어떤지 분석해주세요. 현재 사용자의 감정 상태가 어떤지, 0부터 100까지 척도를 측정하여 답변에 포함해주세요.\n 답변 양식은 다음과 같습니다.\n ### 답변양식\n 현재 상태: 기쁨\n 척도: 88 \n {당신의 답변}"},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=256
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(f'response: {response}')