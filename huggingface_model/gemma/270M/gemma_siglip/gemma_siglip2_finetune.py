import os
import random
import argparse
import getpass
import torch
import torch.nn as nn
import climage
from datasets import load_dataset
from huggingface_hub import login, get_token
from transformers import (
    AutoModelForCausalLM, 
    SiglipVisionModel,
    Trainer, 
    TrainingArguments, 
    AutoTokenizer, 
    AutoImageProcessor, 
    TrainerCallback
)

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
OUTPUT_DIR = "./gemma-nano-siglip2"
BATCH_SIZE = 2
ACCUMULATION_STEPS = 8
LR = 5e-4
EPOCHS = 1  # 1 full pass over 30,000 images is perfect
MODEL_WEIGHTS = f"{OUTPUT_DIR}/best_pytorch_model.bin"
NUM_TESTS = 5  
OUTPUT_IMG_DIR = "./validation_images" 
TEST_SIZE = 400
SEED = 42

# ==========================================
# HUGGING FACE AUTHENTICATION
# ==========================================
def authenticate_hf():
    current_token = get_token()
    
    if current_token is None:
        print("⚠️ Hugging Face token not found in environment or local cache.")
        print("Gemma 3 requires a Hugging Face token to download.")
        hf_token = getpass.getpass("Please enter your HF token (input will be hidden): ")
        
        # This logs you in AND saves the token persistently
        login(token=hf_token)
        print("✅ Token saved locally. You won't be prompted again on this machine.")
    else:
        print("✅ Hugging Face token found! Already authenticated.")
        # Optional: login again just to ensure the session is active
        login(token=current_token)

authenticate_hf()

# ==========================================
# MODEL DEFINITION
# ==========================================
class Gemma3Nano(nn.Module):
    def __init__(self):
        super().__init__()
        print("Loading Vision: SigLIP 2 Base...")
        self.vision_tower = SiglipVisionModel.from_pretrained("google/siglip2-base-patch16-224")
        self.vision_tower.requires_grad_(False)
        self.vision_hidden_size = self.vision_tower.config.hidden_size

        print("Loading Text: Gemma 3 270M...")
        self.llm = AutoModelForCausalLM.from_pretrained("google/gemma-3-270m-it")
        self.text_hidden_size = self.llm.config.hidden_size
        self.llm.requires_grad_(False)

        self.projector = nn.Sequential(
            nn.Linear(self.vision_hidden_size, self.text_hidden_size),
            nn.GELU(),
            nn.Linear(self.text_hidden_size, self.text_hidden_size),
            nn.LayerNorm(self.text_hidden_size)
        )

    def forward(self, input_ids, pixel_values, attention_mask=None, labels=None):
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            with torch.no_grad():
                vision_outputs = self.vision_tower(pixel_values=pixel_values)
                image_features = vision_outputs.last_hidden_state

            image_embeds = self.projector(image_features)

            # 1. Sandwich Embeddings
            bos_embeds = inputs_embeds[:, :1, :]
            rest_embeds = inputs_embeds[:, 1:, :]
            inputs_embeds = torch.cat([bos_embeds, image_embeds, rest_embeds], dim=1)

            # 2. Sandwich Attention Mask
            if attention_mask is not None:
                bos_mask = attention_mask[:, :1]
                rest_mask = attention_mask[:, 1:]
                dummy_mask = torch.ones((attention_mask.shape[0], image_embeds.shape[1]), device=attention_mask.device, dtype=attention_mask.dtype)
                attention_mask = torch.cat([bos_mask, dummy_mask, rest_mask], dim=1)

            # 3. Sandwich Labels
            if labels is not None:
                bos_labels = labels[:, :1]
                rest_labels = labels[:, 1:]
                dummy_labels = torch.full((labels.shape[0], image_embeds.shape[1]), -100, device=labels.device, dtype=labels.dtype)
                labels = torch.cat([bos_labels, dummy_labels, rest_labels], dim=1)

        return self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels, return_dict=True)

# ==========================================
# TRAINING COMPONENTS
# ==========================================
class SaveBestModelCallback(TrainerCallback):
    def __init__(self, output_dir):
        self.best_loss = float('inf')
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.save_path = f"{self.output_dir}/best_pytorch_model.bin"

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        eval_loss = float(metrics.get('eval_loss', float('inf')))
        if eval_loss < self.best_loss:
            print(f"\n🌟 New best validation loss: {eval_loss:.4f}. Saving!")
            self.best_loss = eval_loss
            torch.save(kwargs['model'].state_dict(), self.save_path)

def prepare_data(batch, tokenizer, processor):
    images = [x.convert("RGB") for x in batch['image']]
    pixel_values = processor(images=images, return_tensors="pt").pixel_values

    prompt_prefix = "<start_of_turn>user\nDescribe this image.<end_of_turn>\n<start_of_turn>model\n"

    input_ids_list = []
    attention_mask_list = [] 
    labels_list = []

    prompt_ids = tokenizer(prompt_prefix, add_special_tokens=False).input_ids
    prompt_mask_len = 1 + len(prompt_ids)

    tokenizer.padding_side = "right"

    for t in batch['text']:
        full_text = f"{prompt_prefix}{t}<end_of_turn>"

        tokens = tokenizer(full_text, truncation=True, max_length=128, padding="max_length", return_tensors="pt")
        ids = tokens.input_ids[0]
        mask = tokens.attention_mask[0] 
        lbls = ids.clone()

        lbls[:prompt_mask_len] = -100
        lbls[mask == 0] = -100 

        input_ids_list.append(ids)
        attention_mask_list.append(mask)
        labels_list.append(lbls)

    return {
        "input_ids": torch.stack(input_ids_list),
        "attention_mask": torch.stack(attention_mask_list), 
        "pixel_values": pixel_values,
        "labels": torch.stack(labels_list)
    }

def train_nano():
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m-it")
    tokenizer.pad_token = tokenizer.eos_token
    processor = AutoImageProcessor.from_pretrained("google/siglip2-base-patch16-224")
    model = Gemma3Nano()

    print("Loading 30k COCO Dataset for Real Pre-Training...")
    dataset = load_dataset("sayakpaul/coco-30-val-2014", split="train")
    split_ds = dataset.train_test_split(test_size=TEST_SIZE, seed=SEED)
    train_ds = split_ds['train']
    eval_ds = split_ds['test']

    def collate_fn(batch):
        texts = []
        for item in batch:
            val = item.get('text', item.get('caption', 'A photo.'))
            if isinstance(val, list):
                val = val[0]
            texts.append(val)
            
        return prepare_data({
            'image': [item['image'] for item in batch],
            'text': texts
        }, tokenizer, processor)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=ACCUMULATION_STEPS,
        learning_rate=LR,
        bf16=True,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=250,
        save_strategy="no",
        remove_unused_columns=False,
        report_to="none"
    )

    trainer = Trainer(
        model=model, 
        args=training_args, 
        train_dataset=train_ds, 
        eval_dataset=eval_ds, 
        data_collator=collate_fn, 
        callbacks=[SaveBestModelCallback(OUTPUT_DIR)]
    )
    trainer.train()

# ==========================================
# VALIDATION COMPONENTS
# ==========================================
def run_batch_validation():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
    
    print("Loading tokenizer, processor, and model skeleton...")
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m-it")
    processor = AutoImageProcessor.from_pretrained("google/siglip2-base-patch16-224")
    
    model = Gemma3Nano()
    
    print(f"Loading trained weights from {MODEL_WEIGHTS}...")
    try:
        state_dict = torch.load(MODEL_WEIGHTS, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
    except FileNotFoundError:
        print(f"\nError: Could not find {MODEL_WEIGHTS}. Did training finish?")
        return

    model.to(device, dtype=torch.bfloat16)
    model.eval()

    print("Fetching the unseen COCO validation data...")
    dataset = load_dataset("sayakpaul/coco-30-val-2014", split="train")
    split_ds = dataset.train_test_split(test_size=TEST_SIZE, seed=SEED)
    eval_ds = split_ds['test']
    
    indices = random.sample(range(len(eval_ds)), NUM_TESTS)
    
    print("\n============================================================")
    print(f"   STARTING BATCH VALIDATION ({NUM_TESTS} UNSEEN IMAGES)")
    print("============================================================")
    
    for i, idx in enumerate(indices):
        sample = eval_ds[idx]
        image = sample['image'].convert("RGB")
        
        ground_truth = sample.get('text', sample.get('caption', 'A photo.'))
        if isinstance(ground_truth, list): ground_truth = ground_truth[0]
        
        save_path = os.path.join(OUTPUT_IMG_DIR, f"val_image_{i+1}.jpg")
        image.save(save_path)
        
        print(f"\n--- Image {i+1}/{NUM_TESTS} ---")
        try:
            cli_image = climage.convert(save_path, is_unicode=True, width=60)
            print(cli_image)
        except Exception as e:
            print(f"[Could not render CLI image: {e}]")
            
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device, dtype=torch.bfloat16)

        prompt = "<start_of_turn>user\nDescribe this image.<end_of_turn>\n<start_of_turn>model\n"
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        
        with torch.no_grad():
            vision_outputs = model.vision_tower(pixel_values=pixel_values)
            image_embeds = model.projector(vision_outputs.last_hidden_state)
            text_embeds = model.llm.get_input_embeddings()(input_ids)

            bos_embeds = text_embeds[:, :1, :]
            rest_embeds = text_embeds[:, 1:, :]
            inputs_embeds = torch.cat([bos_embeds, image_embeds, rest_embeds], dim=1)

            attention_mask = torch.ones(inputs_embeds.shape[:2], device=device, dtype=torch.long)

            output_ids = model.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=40,
                do_sample=False,
                repetition_penalty=1.15,
                pad_token_id=tokenizer.eos_token_id
            )
            
        prediction = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        
        print(f"💾 Saved to      : {save_path}")
        print(f"✅ Ground Truth : {ground_truth}")
        print(f"🤖 Prediction   : {prediction}")
        print("-" * 60)

# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or Validate Gemma3Nano")
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["train", "validate", "both"], 
        default="both", 
        help="Select whether to run the training phase, validation phase, or both."
    )
    args = parser.parse_args()

    if args.mode in ["train", "both"]:
        print("\n>>> INITIATING TRAINING PHASE <<<")
        train_nano()
        
    if args.mode in ["validate", "both"]:
        print("\n>>> INITIATING VALIDATION PHASE <<<")
        run_batch_validation()
