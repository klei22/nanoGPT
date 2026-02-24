# image_chat.py
import os
import torch
import argparse
from PIL import Image
from transformers import AutoTokenizer, AutoImageProcessor

from gemma_siglip2_finetune import Gemma3Nano

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_WEIGHTS = "./gemma-nano-siglip2/best_pytorch_model.bin"

def chat_with_image(image_path):
    # 1. Standard Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Initializing on {device.upper()}...")

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m-it")
    processor = AutoImageProcessor.from_pretrained("google/siglip2-base-patch16-224")

    model = Gemma3Nano()
    
    try:
        state_dict = torch.load(MODEL_WEIGHTS, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
    except FileNotFoundError:
        print(f"❌ Error: Could not find weights at {MODEL_WEIGHTS}.")
        return

    model.to(device, dtype=torch.bfloat16)
    model.eval()

    # 2. Process the Image ONCE
    print(f"🖼️  Analyzing image: {image_path}...")
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device, dtype=torch.bfloat16)

    with torch.no_grad():
        # We process the vision tower outside the loop to save massive amounts of compute
        vision_outputs = model.vision_tower(pixel_values=pixel_values)
        image_embeds = model.projector(vision_outputs.last_hidden_state)

    # 3. Initialize Conversation Memory
    history = ""
    print("\n💬 Chat session started! (Type 'quit' or 'exit' to end)")
    print("-" * 60)

    # 4. The Interactive Loop
    while True:
        user_input = input("\nYou: ")
        
        # Exit condition
        if user_input.lower() in ['quit', 'exit']:
            print("Ending chat. Goodbye! 👋")
            break
            
        # Append the new user turn to the conversation history
        history += f"<start_of_turn>user\n{user_input}<end_of_turn>\n<start_of_turn>model\n"
        
        # Tokenize the ENTIRE history so the model remembers past turns
        input_ids = tokenizer(history, return_tensors="pt").input_ids.to(device)

        with torch.no_grad():
            text_embeds = model.llm.get_input_embeddings()(input_ids)

            # The Sandwich: We always insert the image right after the BOS token
            # of the total conversation history.
            bos_embeds = text_embeds[:, :1, :]
            rest_embeds = text_embeds[:, 1:, :]
            inputs_embeds = torch.cat([bos_embeds, image_embeds, rest_embeds], dim=1)

            attention_mask = torch.ones(inputs_embeds.shape[:2], device=device, dtype=torch.long)

            # Generate the response
            output_ids = model.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=100,      # Increased to allow for longer conversational replies
                do_sample=True,          # Turned ON for chat (makes conversation feel more natural)
                temperature=0.7,         # Controls creativity
                repetition_penalty=1.15, 
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode the newly generated text
        prediction = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        print(f"🤖 Model: {prediction}")
        
        # Crucial Step: Append the model's response to the history so it remembers what IT said!
        history += f"{prediction}<end_of_turn>\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chat with your custom Vision-Language Model")
    parser.add_argument("image_path", type=str, help="Path to the image file")
    args = parser.parse_args()
    
    chat_with_image(args.image_path)
