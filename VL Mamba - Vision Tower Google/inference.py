import torch
from PIL import Image
from transformers import AutoTokenizer, AutoFeatureExtractor, AutoModel
from transformers import MambaForCausalLM

# List of GPUs to use
device_ids = [3]  # Specify device indices you want to use
device = torch.device(f"cuda:{device_ids[0]}" if torch.cuda.is_available() else "cpu")

# Load the vision model (e.g., ViT) and feature extractor
vision_model = AutoModel.from_pretrained("/scratch/faaraan/mamba-chat/googlevit-base-patch16-224")
vision_model.eval().to(device)
feature_extractor = AutoFeatureExtractor.from_pretrained("/scratch/faaraan/mamba-chat/googlevit-base-patch16-224")

# Load the tokenizer and Mamba model for causal LM
tokenizer = AutoTokenizer.from_pretrained("/scratch/faaraan/mamba-chat/mamba-2.8b-hf")
mamba_llm = MambaForCausalLM.from_pretrained("/scratch/faaraan/mamba-chat/mamba-2.8b-hf").to(device)

# Create a linear layer to map visual embeddings to the same size as text embeddings
visual_to_text_proj = torch.nn.Linear(768, 2560).to(device)

# Load model checkpoint
optimizer = torch.optim.AdamW(mamba_llm.parameters(), lr=5e-5)

def load_checkpoint(checkpoint_path, mamba_llm, optimizer):
    checkpoint = torch.load(checkpoint_path)
    mamba_llm.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Checkpoint loaded from {checkpoint_path}, starting from epoch {epoch}")
    return epoch

# Load your checkpoint
load_checkpoint('/scratch/faaraan/mamba-chat/VL-Mamba-Vision-Tower-Google/checkpoints/checkpoint_epoch_60.pt', mamba_llm, optimizer)

# Function to generate response
def generate_response(image_url, text_prompt, max_length=150, temperature=0.8, top_k=50, top_p=0.95, num_beams=1, do_sample=True):
    # Step 1: Prepare the image
    image = Image.open(image_url)
    image_inputs = feature_extractor(images=image, return_tensors="pt").to(device)  # Preprocess image and move to GPU

    # Step 2: Extract visual features
    with torch.no_grad():
        visual_features = vision_model(**image_inputs).last_hidden_state  # Shape: [1, 196, 768]

    # Step 3: Project visual features to match the dimensionality of text embeddings
    visual_output = visual_features.mean(dim=1)  # Shape: [1, 768]
    visual_output_proj = visual_to_text_proj(visual_output)  # Shape: [1, 2560]

    # Step 4: Tokenize the text input
    text_inputs = tokenizer(text_prompt, return_tensors="pt", padding=True, truncation=True).to(device)  # Move to GPU

    # Step 5: Embed visual features as a token
    visual_token = visual_output_proj.unsqueeze(1)  # Shape: [1, 1, 2560]

    # Step 6: Combine visual token with text embeddings
    text_embeddings = mamba_llm.get_input_embeddings()(text_inputs["input_ids"])
    combined_embeddings = torch.cat([visual_token, text_embeddings], dim=1)  # Shape: [1, N+1, 2560]

    # Ensure attention mask is set to avoid warnings
    attention_mask = torch.cat([torch.ones(visual_token.shape[:-1], device=device), text_inputs["attention_mask"]], dim=1)

    # Step 7: Forward pass through Mamba LLM with combined embeddings
    outputs = mamba_llm.generate(
        inputs_embeds=combined_embeddings,
        attention_mask=attention_mask,     # Pass the attention mask
        max_length=max_length,             # Control the maximum length of the response
        temperature=temperature,           # Controls the randomness of predictions
        top_k=top_k,                       # Top-k sampling
        top_p=top_p,                       # Nucleus sampling
        num_beams=num_beams,               # Beam search
        do_sample=do_sample,               # Enable sampling for diverse outputs
        no_repeat_ngram_size=2,            # Prevent repeating n-grams
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    # Step 8: Generate text from the output logits
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return generated_text


# Test the function with an image and prompt
# image_url = '/scratch/faaraan/mamba-chat/week_02_page_002.png'
image_url = '/scratch/faaraan/LLaVAData/images/week_02/week_02_page_016.png'

text_prompt = "How does the analysis of decision boundaries extend to higher-dimensional spaces?"
response = generate_response(image_url, text_prompt, max_length=200, temperature=0.9, top_k=50, top_p=0.95, num_beams=3, do_sample=True)
print("Image:", image_url)
print("Question:", text_prompt)
print("Response:", response)
