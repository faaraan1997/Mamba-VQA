import torch
import torch.nn as nn
from torchvision import transforms
from transformers import AutoTokenizer, AutoModelForImageClassification, MambaForCausalLM
from PIL import Image
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

    Args:
        image_path (str): Path to the input image.
        text_prompt (str): The text prompt/question.
        model_saved (str): Path to the saved model checkpoint.
        max_length (int): Maximum length of the generated response.
        temperature (float): Sampling temperature.
        top_k (int): Top-k sampling.
        top_p (float): Top-p (nucleus) sampling.
        num_beams (int): Number of beams for beam search.
        do_sample (bool): Whether to use sampling.
        device (str): Device to run the models on ('cuda:0' or 'cpu').

    Returns:
        str: The generated response.
    """
    # -----------------------------
    # 2.1. Initialize and Load Models
    # -----------------------------
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

    # -----------------------------
    # 2.2. Define Image Transformations
    # -----------------------------
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match vision model input
        transforms.ToTensor(),          # Convert PIL Image to Tensor
        transforms.Normalize(           # Normalize based on ImageNet stats or model requirements
            mean=[0.485, 0.456, 0.406],  # Example means
            std=[0.229, 0.224, 0.225]    # Example stds
        )
    ])

    # -----------------------------
    # 2.3. Load and Preprocess the Image
    # -----------------------------
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise IOError(f"Error loading image: {e}")

    image_tensor = transform(image).unsqueeze(0).to(device)  # Shape: [1, 3, 224, 224]

    # -----------------------------
    # 2.4. Extract Visual Features
    # -----------------------------
    with torch.no_grad():
        visual_output = vision_model(image_tensor)
        if isinstance(visual_output, dict):
            visual_features = visual_output.get('logits')  # Shape: [1, 1000]
        else:
            visual_features = visual_output  # Adjust if necessary

    # Pass visual features through MMC
    visual_embeddings = mmc(visual_features)  # Shape: [1, 2560]
    visual_token = visual_embeddings.unsqueeze(1)  # Shape: [1, 1, 2560]

    # -----------------------------
    # 2.5. Tokenize the Text Prompt
    # -----------------------------
      # Modify the prompt to guide the model toward a 250-word summary
    text_prompt += " Please summarize the analysis in approximately 250 words or less, organizing the summary into paragraphs."

    inputs = tokenizer(text_prompt, return_tensors="pt", truncation=True, padding=True)
    input_ids = inputs['input_ids'].to(device)  # Shape: [1, seq_len]
    attention_mask = inputs['attention_mask'].to(device)  # Shape: [1, seq_len]

    # Get text embeddings from LLM
    text_embeddings = mamba_llm.get_input_embeddings()(input_ids)  # Shape: [1, seq_len, 2560]

    # -----------------------------
    # 2.6. Combine Visual and Text Embeddings
    # -----------------------------
    combined_embeddings = torch.cat([visual_token, text_embeddings], dim=1)  # Shape: [1, seq_len + 1, 2560]

    # Create combined attention mask
    visual_attention = torch.ones((attention_mask.size(0), 1), device=device, dtype=attention_mask.dtype)  # Shape: [1,1]
    combined_attention_mask = torch.cat([visual_attention, attention_mask], dim=1)  # Shape: [1, seq_len + 1]

    # -----------------------------
    # 2.7. Generate Response
    # -----------------------------
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
            pad_token_id=tokenizer.eos_token_id  # Ensure padding is handled correctly
        )

    # Decode the generated tokens
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return generated_text

# -----------------------------
# 3. Perform Inference
# -----------------------------
if __name__ == "__main__":
    # Path to the saved checkpoint
    model_saved = "/scratch/faaraan/mamba-chat/Mamba - Vision Tower Mamba VL/checkpoints/epoch_new_2.pt"

    # Input image and prompt
    image_path = "/scratch/faaraan/LLaVAData/images/week_02/week_02_page_016.png"
    text_prompt = "How does the analysis of decision boundaries extend to higher-dimensional spaces?"

    # Set device to CUDA device 0
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Check if CUDA device 0 is available
    if device == 'cuda:0':
        if not torch.cuda.is_available():
            raise EnvironmentError("CUDA is not available, but 'cuda:0' was requested.")
        if torch.cuda.device_count() < 1:
            raise EnvironmentError("No CUDA devices available, but 'cuda:0' was requested.")
    else:
        print("CUDA device not available. Using CPU.")

    # Generate response
    try:
        response = generate_response(
            image_path=image_path,
            text_prompt=text_prompt,
            model_saved=model_saved,
            max_length=200,
            temperature=0.9,
            top_k=50,
            top_p=0.95,
            num_beams=3,
            do_sample=True,
            device=device
        )

        # Print the results
        print("Image:", image_path)
        print("Question:", text_prompt)
        print("Response:", response)
    except Exception as e:
        print(f"An error occurred during inference: {e}")
