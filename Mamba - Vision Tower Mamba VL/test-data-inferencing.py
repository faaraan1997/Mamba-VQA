import torch
import torch.nn as nn
from torchvision import transforms
from transformers import AutoTokenizer, AutoModelForImageClassification, MambaForCausalLM
from PIL import Image
import json
import os

# -----------------------------
# 1. Define the MultiModal Connector (MMC)
# -----------------------------
class MMC(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MMC, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 2 * input_dim),
            nn.GELU(),
            nn.Linear(2 * input_dim, output_dim)
        )

    def forward(self, visual_features):
        return self.mlp(visual_features)

# -----------------------------
# 2. Define the generate_response Function
# -----------------------------
def generate_response(image_path, text_prompt, model_saved, 
                      max_length=200, temperature=0.9, top_k=50, 
                      top_p=0.95, num_beams=3, do_sample=True, device='cuda:0'):
    """
    Generates a response based on the input image and text prompt.
    """
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("/scratch/faaraan/mamba-chat/mamba-2.8b-hf")

    # Load the vision model
    vision_model = AutoModelForImageClassification.from_pretrained(
        "nvidia/MambaVision-T-1K", trust_remote_code=True
    )
    vision_model.to(device)
    vision_model.eval()

    # Initialize MMC and load state_dict
    mmc = MMC(input_dim=1000, output_dim=2560).to(device)

    # Load the language model
    mamba_llm = MambaForCausalLM.from_pretrained("/scratch/faaraan/mamba-chat/mamba-2.8b-hf").to(device)
    mamba_llm.eval()

    # Load the checkpoint
    if not os.path.isfile(model_saved):
        raise FileNotFoundError(f"Checkpoint file not found at {model_saved}")

    checkpoint = torch.load(model_saved, map_location=device)
    mmc.load_state_dict(checkpoint['mmc_state_dict'])
    mamba_llm.load_state_dict(checkpoint['mamba_llm_state_dict'])
    print(f"Loaded checkpoint from {model_saved}")

    # Define Image Transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load and Preprocess the Image
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise IOError(f"Error loading image: {e}")

    image_tensor = transform(image).unsqueeze(0).to(device)

    # Extract Visual Features
    with torch.no_grad():
        visual_output = vision_model(image_tensor)
        visual_features = visual_output.get('logits') if isinstance(visual_output, dict) else visual_output

    # Pass visual features through MMC
    visual_embeddings = mmc(visual_features)
    visual_token = visual_embeddings.unsqueeze(1)

    # Tokenize the Text Prompt
    text = f"{text_prompt}? "
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # Get text embeddings from LLM
    text_embeddings = mamba_llm.get_input_embeddings()(input_ids)

    # Combine Visual and Text Embeddings
    combined_embeddings = torch.cat([visual_token, text_embeddings], dim=1)
    visual_attention = torch.ones((attention_mask.size(0), 1), device=device, dtype=attention_mask.dtype)
    combined_attention_mask = torch.cat([visual_attention, attention_mask], dim=1)

    # Generate Response
    with torch.no_grad():
        output_ids = mamba_llm.generate(
            inputs_embeds=combined_embeddings,
            attention_mask=combined_attention_mask,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_beams=num_beams,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text

# -----------------------------
# 3. Process the JSON Data and Generate Responses
# -----------------------------
def process_data(json_file, model_saved, output_file, device='cuda:0'):
    """
    Process the input JSON file, generate predictions, and save results.
    """
    with open(json_file, 'r') as f:
        data = json.load(f)

    results = []
    for entry in data:
        image_path = entry['image']
        text_prompt = next((conv['value'] for conv in entry['conversations'] if conv['from'] == 'human'), None)
        ground_truth = next((conv['value'] for conv in entry['conversations'] if conv['from'] == 'gpt'), None)

        # Clean the text_prompt by removing "<image>\n"
        if text_prompt:
            text_prompt = text_prompt.replace("<image>\n", "").strip()
            
        if not text_prompt:
            print(f"No text prompt found for image {image_path}, skipping.")
            continue

        try:
            predicted_response = generate_response(
                image_path=image_path,
                text_prompt=text_prompt,
                model_saved=model_saved,
                device=device
            )
            result = {
                "Question": text_prompt,
                "Image_path": image_path,
                "Ground Truth": ground_truth,
                "Predicted": predicted_response
            }
            results.append(result)
            print(f"Processed image {image_path}: {predicted_response}")
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    # Save results to JSON file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_file}")

# -----------------------------
# 4. Main Function to Execute the Code
# -----------------------------
if __name__ == "__main__":
    # Paths and device setup
    model_saved = "/scratch/faaraan/mamba-chat/Mamba - Vision Tower Mamba VL/checkpoints/epoch_50.pt"
    json_file = "/scratch/faaraan/LLaVAData/data_llava_data_week_1_to_8.json"
    output_file = "/scratch/faaraan/LLaVAData/predictions_week_1_to_8.json"
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Check if CUDA is available
    if device == 'cuda:7':
        if not torch.cuda.is_available():
            raise EnvironmentError("CUDA is not available, but 'cuda:7' was requested.")
        if torch.cuda.device_count() < 1:
            raise EnvironmentError("No CUDA devices available, but 'cuda:7' was requested.")
    else:
        print("CUDA device not available. Using CPU.")

    # Process data and generate responses
    try:
        process_data(json_file=json_file, model_saved=model_saved, output_file=output_file, device=device)
    except Exception as e:
        print(f"An error occurred during processing: {e}")
