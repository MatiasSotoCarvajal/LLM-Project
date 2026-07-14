from backend.llama_server import run_single

MODELS: list[str] = ["unsloth/gemma-4-E4B-it-GGUF",
                     "unsloth/gemma-4-E2B-it-GGUF",
                     "unsloth/Qwen3.5-9B-GGUF",
                     "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
                     ]

Q_TYPE: str = "tq4_1s"
Q_MODELS: list[str] = [f"{model}-{Q_TYPE}" for model in MODELS]

model_id = Q_MODELS[0]
print(model_id)
prompt = "What model are you?"
system_prompt = "Respond with the most detailed response."
r = run_single(model_id, prompt, system_prompt)
print(r)


#if __name__ == "__main__":
    #start()