from arzlm.infer import generate, load_inference_checkpoint

loaded = load_inference_checkpoint("Kymaris/ArzLM-300M-Base", device="cuda", dtype="bf16")
print("params", loaded.parameters, "ctx", loaded.config.block_size)
print(generate(loaded, "The capital of France is", max_new_tokens=32)["completion"])
