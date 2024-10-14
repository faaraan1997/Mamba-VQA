from transformers import AutoTokenizer, AutoModelForImageClassification, MambaForCausalLM
from PIL import Image
import requests
import torch
from timm.data.transforms_factory import create_transform

# Load the vision model (MambaVision)
vision_model = AutoModelForImageClassification.from_pretrained("nvidia/MambaVision-T-1K", trust_remote_code=True)
vision_model.eval().cuda()

# Define the MultiModal Connector (MMC) with output_dim matching the LLM embedding size
class MMC(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MMC, self).__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 2 * input_dim),
            torch.nn.GELU(),
            torch.nn.Linear(2 * input_dim, output_dim)
        )

    def forward(self, visual_features):
        return self.mlp(visual_features)

# Adjust input_dim to match the output size of the vision model (1000)
mmc = MMC(input_dim=1000, output_dim=2560).cuda()

# Load the Mamba LLM and tokenizer
tokenizer = AutoTokenizer.from_pretrained("/scratch/faaraan/mamba-chat/mamba-2.8b-hf")
mamba_llm = MambaForCausalLM.from_pretrained("/scratch/faaraan/mamba-chat/mamba-2.8b-hf").cuda()

# Function to generate a response
def generate_response(image_url, text_prompt):
    # Step 1: Prepare the image
    # image = Image.open(requests.get(image_url, stream=True).raw)
    image = Image.open(image_url).convert("RGB")
    transform = create_transform(input_size=(3, 224, 224), is_training=False)
    image_tensor = transform(image).unsqueeze(0).cuda()

    # Step 2: Extract visual features from the MambaVision model
    with torch.no_grad():
        visual_output = vision_model(image_tensor)
        if isinstance(visual_output, dict):
            visual_features = visual_output.get('logits')  # Extract logits
        else:
            visual_features = visual_output

    # Step 3: Pass visual features through the MMC
    visual_output = mmc(visual_features)  # Shape: [1, 2560]

    # Step 4: Tokenize the text input
    input_ids = tokenizer(text_prompt, return_tensors="pt")["input_ids"].cuda()

    # Step 5: Embed visual features as a token
    visual_token = visual_output.unsqueeze(1)  # Shape: [1, 1, 2560]

    # Step 6: Combine visual token with text embeddings
    text_embeddings = mamba_llm.get_input_embeddings()(input_ids)
    combined_embeddings = torch.cat([visual_token, text_embeddings], dim=1)  # Shape: [1, N+1, 2560]

    # Step 7: Forward pass through Mamba LLM with combined embeddings
    outputs = mamba_llm.generate(
        inputs_embeds=combined_embeddings,
        max_new_tokens=150,  # Increase this to generate more tokens
        min_new_tokens=50,   # Optional: Ensure minimum length
        do_sample=True,      # Enable sampling for more diverse responses
        top_p=0.95,          # Nucleus sampling
        temperature=0.8,     # Adjust temperature for randomness
    )

    # Step 8: Decode the generated tokens into text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return generated_text

# Test the function with an image and prompt
image_url = '/scratch/faaraan/LLaVAData/images/week_02/week_02_page_016.png'
text_prompt = "How does the analysis of decision boundaries extend to higher-dimensional spaces?"
response = generate_response(image_url, text_prompt)
print("Response:", response)