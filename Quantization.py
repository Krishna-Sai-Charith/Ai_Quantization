# =======================
# Colab-specific code (COMMENTED OUT for VS Code use)
# =======================
# !pip install -q --upgrade torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124
# !pip install -q requests bitsandbytes==0.46.0 transformers==4.48.3 accelerate==1.3.0
# from google.colab import userdata
# hf_token = userdata.get('HF_TOKEN')
# from huggingface_hub import login
# login(hf_token, add_to_git_credential=True)

# =======================
# VS Code / Local Setup
# =======================
# 1. Install dependencies manually before running:
#    pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124
#    pip install requests bitsandbytes==0.46.0 transformers==4.48.3 accelerate==1.3.0 huggingface_hub
#
# 2. Store your Hugging Face token in an environment variable:
#    Linux/Mac: export HF_TOKEN="your_token_here"
#    Windows (PowerShell): setx HF_TOKEN "your_token_here"

import os
import torch
import gc
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig

# Load Hugging Face token from environment variable (more secure than hardcoding)
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("HF_TOKEN environment variable not set. Please set it before running.")

# Login to Hugging Face
login(hf_token, add_to_git_credential=True)

# =======================
# Model Choices
# =======================
LLAMA = "meta-llama/Meta-Llama-3.1-8B-Instruct"
PHI3 = "microsoft/Phi-3-mini-4k-instruct"
GEMMA2 = "google/gemma-2-2b-it"
QWEN2 = "Qwen/Qwen2-7B-Instruct"
MIXTRAL = "mistralai/Mixtral-8x7B-Instruct-v0.1"

# Example messages
messages = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Tell a light-hearted joke for a room of Data Scientists"}
]

# =======================
# Quantization Config
# =======================
# This reduces memory usage by loading the model in 4-bit precision.
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)

# =======================
# Generation Function
# =======================
def generate(model_name, messages):
    """
    Loads the model in 4-bit quantization and generates a response for given chat messages.

    Args:
        model_name (str): Hugging Face model ID.
        messages (list): List of dicts with 'role' and 'content' keys.

    Steps:
        1. Load tokenizer
        2. Prepare prompt using chat template
        3. Load quantized model
        4. Generate text with streaming output
        5. Free GPU memory after generation
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True
    ).to("cuda")

    streamer = TextStreamer(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=quant_config
    )

    outputs = model.generate(inputs, max_new_tokens=80, streamer=streamer)

    # Free memory after generation
    del model, inputs, tokenizer, outputs, streamer
    gc.collect()
    torch.cuda.empty_cache()

# =======================
# Example Usage
# =======================
generate(PHI3, messages)
# generate(LLAMA, messages)
# generate(GEMMA2, messages)
# generate(QWEN2, messages)
# generate(MIXTRAL, messages)
