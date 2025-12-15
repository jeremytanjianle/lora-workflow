# LoRA workflow on Runpod 
Workflow for friends and collaborators to adapt.

## Setup 

1. Set up environment: `uv sync`
2. Ensure Ollama is installed, with `qwen2.5:14b` model downloaded: `ollama pull qwen2.5:14b`
3. Add huggingface token to `.env`, to use gated models

## Workflow

### 1. Data generation to finetune model

Set our in-domain dataset in `./data/instruction.json`.   
If needed, we can parse PDFs to generate some QA pairs.  

1. Parse source PDFs: `uv run 1a_parse_pdf.py`
2. Generate QA pairs: `uv run 1b_generate_qa_pairs.py`
3. Clean up dataset quality: `uv run 1c_data_quality.py`  

Otherwise, use `./data/original` and skip to the next step.

### 2. Finetune on runpod

1. Set up environment: `uv sync --group train`
2. Run training script: `uv run 2a_train.py`
3. Save the model artifacts from `./final_model/`

### 3. Load finetuning on ollama
1. Modify `Modelfile`, namely the `ADAPTER` clause
2. Create the model in ollama: `ollama create tm1bud -f modelfile_adapter`

If ollama fails to create the model for pathing reasons (in my case, it seems to a Windows-specific problem), merge the model and create from the merged model directly. 
1. `uv run 2b_merge_lora.py`
2. `ollama create tm1bud -f modelfile_from_merged`

Run finetuned model on ollama
`ollama run tm1bud --verbose`

## References
[Source tutorial](https://youtu.be/D3pXSkGceY0), which I've adapted for greater ease of understanding  
[LLaMA docs](https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_2/)  
[docling](https://www.docling.ai/)  