from transformers import AutoConfig, AutoModel, AutoTokenizer

# Upload model artifacts saved with `save_pretrained(...)`
config = AutoConfig.from_pretrained("gpt2-custom")
config.push_to_hub("custom_gpt2")

model = AutoModel.from_pretrained("gpt2-custom")
model.push_to_hub("custom_gpt2")

tokenizer = AutoTokenizer.from_pretrained("gpt2-custom")
tokenizer.push_to_hub("custom_gpt2")

# If you have only a tokenizer.json file, use:
#   python huggingface_model/upload_tokenizer_from_hf_json.py \
#       --tokenizer_json path/to/tokenizer.json \
#       --repo_id your-hf-user/your-tokenizer-repo
