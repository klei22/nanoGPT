import os
import torch
import argparse
from PIL import Image
from transformers import AutoTokenizer, AutoImageProcessor

# --- IMPORT ARCHITECTURE ---
from gemma_siglip2_finetune import Gemma3Nano

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_WEIGHTS = "./gemma-nano-siglip2/best_pytorch_model.bin"

def run_inference(image_path):
    """
    Runs inference using the custom Gemma-3-Nano + SigLIP-2 model.
    Takes a path to an image and outputs a generated description.
    """
    # 1. Device Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Using device: {device.upper()}")

    # 2. Check for Weights
    if not os.path.exists(MODEL_WEIGHTS):
        print(f"❌ Error: Could not find weights at {MODEL_WEIGHTS}.")
        print("Did you finish running the training script?")
        return

    # 3. Load Processors
    print("📦 Loading tokenizer and image processor...")
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m-it")
    processor = AutoImageProcessor.from_pretrained("google/siglip2-base-patch16-224")

    # 4. Load Model Architecture & Weights
    print("🧠 Initializing model architecture and loading trained weights...")
    model = Gemma3Nano()
    
    # weights_only=True is a security best practice for loading PyTorch binaries
    state_dict = torch.load(MODEL_WEIGHTS, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    # Move to GPU and set to half-precision (bfloat16) for speed
    model.to(device, dtype=torch.bfloat16)
    model.eval() # Disable dropout and batch norm layers for inference

    # 5. Load and Process Image
    print(f"🖼️  Processing image: {image_path}...")
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"❌ Error loading image {image_path}: {e}")
        return

    # Convert image to the pixel values SigLIP expects
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device, dtype=torch.bfloat16)

    # 6. Prepare the Text Prompt
    # We use the exact instruction template used during training
    prompt = "<start_of_turn>user\nDescribe this image.<end_of_turn>\n<start_of_turn>model\n"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    print("\n✨ Generating description...")
    print("-" * 50)
    
    # 7. The Forward Pass (No gradients needed for inference)
    with torch.no_grad():
        # A. Pass image through SigLIP Vision Tower
        vision_outputs = model.vision_tower(pixel_values=pixel_values)
        
        # B. Project vision features into Gemma's text embedding space
        image_embeds = model.projector(vision_outputs.last_hidden_state)
        
        # C. Get the raw token embeddings for the text prompt
        text_embeds = model.llm.get_input_embeddings()(input_ids)

        # D. THE SANDWICH: Combine Text and Image Embeddings
        # We split the text embeddings at the first token (usually the Beginning of Sentence or BOS token)
        # and insert the image embeddings right after it, before the rest of the text.
        bos_embeds = text_embeds[:, :1, :]
        rest_embeds = text_embeds[:, 1:, :]
        inputs_embeds = torch.cat([bos_embeds, image_embeds, rest_embeds], dim=1)

        # Create a basic attention mask of 1s (pay attention to all tokens in the sandwich)
        attention_mask = torch.ones(inputs_embeds.shape[:2], device=device, dtype=torch.long)

        # E. Auto-Regressive Generation
        output_ids = model.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=40,       # Limit the length of the output description
            do_sample=False,         # Greedy decoding (pick the most likely next word)
            repetition_penalty=1.15, # Prevent the model from stuttering or repeating phrases
            pad_token_id=tokenizer.eos_token_id
        )

    # 8. Decode and Display
    # skip_special_tokens=True removes the <eos> and formatting tags
    prediction = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    print(f"🤖 Prediction: {prediction}\n")

# ==========================================
# CLI EXECUTION
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test your custom Vision-Language Model")
    parser.add_argument("image_path", type=str, help="Path to the image file you want the model to describe")
    args = parser.parse_args()
    
    run_inference(args.image_path)
