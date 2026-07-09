from mlx_lm import load, generate

model, tokenizer = load("/Users/matiassoto/Repositorios/RUB/LLM Project/Models/alexcovo__qwen35-9b-mlx-turboquant-tq3") # type: ignore

prompt = "Write a story about Einstein"
messages = [{"role": "user", "content": prompt}]
prompt = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True
)

text = generate(model, tokenizer, prompt=prompt, verbose=True)

print(text)