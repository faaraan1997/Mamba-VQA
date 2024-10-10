from transformers import AutoTokenizer, AutoModel, MambaForCausalLM
from PIL import Image
import requests
import torch
from timm.data.transforms_factory import create_transform

# Load the vision model (ViT)
vision_model = AutoModel.from_pretrained("/scratch/faaraan/mamba-chat/googlevit-base-patch16-224")
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

# Create a MultiModal Connector with output_dim=2560
mmc = MMC(input_dim=768, output_dim=2560).cuda()

# Load the Mamba LLM and tokenizer
tokenizer = AutoTokenizer.from_pretrained("/scratch/faaraan/mamba-chat/mamba-2.8b-hf")
mamba_llm = MambaForCausalLM.from_pretrained("/scratch/faaraan/mamba-chat/mamba-2.8b-hf").cuda()

# Function to generate a response
def generate_response(image_url, text_prompt):
    # Step 1: Prepare the image
    image = Image.open(image_url)
    transform = create_transform(input_size=(3, 224, 224), is_training=False)
    image_tensor = transform(image).unsqueeze(0).cuda()

    # Step 2: Extract visual features
    with torch.no_grad():
        visual_features = vision_model(image_tensor).last_hidden_state  # Shape: [1, 196, 768]

    # Step 3: Pass visual features through the MMC
    visual_output = mmc(visual_features.mean(dim=1))  # Shape: [1, 2560]

    # Step 4: Tokenize the text input
    input_ids = tokenizer(text_prompt, return_tensors="pt")["input_ids"].cuda()

    # Step 5: Embed visual features as a token
    visual_token = visual_output.unsqueeze(1)  # Shape: [1, 1, 2560]

    # Step 6: Combine visual token with text embeddings
    text_embeddings = mamba_llm.get_input_embeddings()(input_ids)
    combined_embeddings = torch.cat([visual_token, text_embeddings], dim=1)  # Shape: [1, N+1, 2560]

    # Step 7: Forward pass through Mamba LLM with combined embeddings
    outputs = mamba_llm(inputs_embeds=combined_embeddings, return_dict=True)

    # Step 8: Generate text from the output logits
    logits = outputs.logits
    predicted_token_ids = logits.argmax(dim=-1)
    generated_text = tokenizer.decode(predicted_token_ids[0], skip_special_tokens=True)

    return generated_text

# Test the function with an image and prompt
image_url = '/scratch/faaraan/mamba-chat/Bear.jpg'
text_prompt = "Describe this image?"
response = generate_response(image_url, text_prompt)
print("Response:", response)
