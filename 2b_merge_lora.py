from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from loguru import logger
from colorama import Fore
import torch
import os
from dotenv import load_dotenv

load_dotenv()

if __name__ == '__main__':

    base_model = "meta-llama/Llama-3.2-1B"
    lora_model_path = "final_model"
    output_path = "final_model_merged"

    logger.info(f"Loading base model: {base_model}")
    # Load base model without quantization (need full precision for merging)
    base_model_loaded = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        token=os.getenv("HF_TOKEN"),
    )

    logger.info(f"Loading tokenizer: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True,
        token=os.getenv("HF_TOKEN"),
    )

    logger.info(f"Loading LoRA adapters from: {lora_model_path}")
    model_with_lora = PeftModel.from_pretrained(base_model_loaded, lora_model_path)

    logger.info(Fore.YELLOW + "Merging LoRA adapters with base model..." + Fore.RESET)
    merged_model = model_with_lora.merge_and_unload()

    logger.info(f"Saving merged model to: {output_path}")
    merged_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    logger.success(f"Model saved")
