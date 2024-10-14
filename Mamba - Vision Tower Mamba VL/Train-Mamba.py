import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoTokenizer, AutoModelForImageClassification, MambaForCausalLM
from tqdm import tqdm
from PIL import Image
import os

# Define the ImageQADataset
class ImageQADataset(Dataset):
    def __init__(self, json_file, transform):
        with open(json_file, "r") as f:
            self.data = json.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item['image']
        question = item['conversations'][0]['value'].replace("<image>\n", "")  # Clean the question
        answer = item['conversations'][1]['value']

        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")  # Ensure image is in RGB
        image_tensor = self.transform(image)

        return image_tensor, question, answer


# Define custom collate function
def custom_collate_fn(batch):
    images, questions, answers = zip(*batch)
    images = torch.stack(images)  # Stack images into a tensor
    return images, questions, answers


# Load the vision model (MambaVision)
vision_model = AutoModelForImageClassification.from_pretrained("nvidia/MambaVision-T-1K", trust_remote_code=True)
vision_model.eval().cuda()

# Define the MultiModal Connector (MMC)
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

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match vision model input
    transforms.ToTensor(),  # Convert to tensor
])

# Load the dataset
dataset = ImageQADataset("/scratch/faaraan/LLaVAData/data_llava_data_week_1_to_8.json", transform)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=custom_collate_fn)

# Initialize the MMC
mmc = MMC(input_dim=1000, output_dim=2560).cuda()

# Load the Mamba LLM and tokenizer
tokenizer = AutoTokenizer.from_pretrained("/scratch/faaraan/mamba-chat/mamba-2.8b-hf")
mamba_llm = MambaForCausalLM.from_pretrained("/scratch/faaraan/mamba-chat/mamba-2.8b-hf").cuda()

# Training configuration
epochs = 5
optimizer = torch.optim.Adam(mmc.parameters(), lr=1e-5)  # Adjust the optimizer and learning rate as needed

# Create a directory to save the checkpoints
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# Training loop
for epoch in range(1, epochs + 1):
    epoch_loss = 0
    for batch_idx, (images, questions, answers) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")):
        # Move images to device
        images = images.cuda()

        # Step 1: Extract visual features from the vision model
        with torch.no_grad():
            visual_output = vision_model(images)
            if isinstance(visual_output, dict):
                visual_features = visual_output.get('logits')  # Extract logits
            else:
                visual_features = visual_output

        # Step 2: Pass visual features through the MMC
        visual_output = mmc(visual_features)  # Shape: [batch_size, 2560]

        # Step 3: Tokenize questions
        input_ids = tokenizer(questions, return_tensors="pt", padding=True, truncation=True).input_ids.cuda()

        # Step 4: Embed visual features as a token
        visual_token = visual_output.unsqueeze(1)  # Shape: [batch_size, 1, 2560]

        # Step 5: Combine visual token with text embeddings
        text_embeddings = mamba_llm.get_input_embeddings()(input_ids)  # Shape: [batch_size, N, 2560]
        combined_embeddings = torch.cat([visual_token, text_embeddings], dim=1)  # Shape: [batch_size, N+1, 2560]

        # Step 6: Forward pass through Mamba LLM
        outputs = mamba_llm.generate(
            inputs_embeds=combined_embeddings,
            max_new_tokens=150,
            min_new_tokens=50,
            do_sample=True,
            top_p=0.95,
            temperature=0.8,
        )

        # Optionally, compute loss and optimize (requires a proper loss function)
        # For example, if using cross-entropy loss:
        # loss = some_loss_function(outputs, answers)
        # epoch_loss += loss.item()
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

    # Save model checkpoint at the end of each epoch
    checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch}.pt")
    torch.save({
        'epoch': epoch,
        'mmc_state_dict': mmc.state_dict(),
        'mamba_llm_state_dict': mamba_llm.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)

    # Print epoch loss if computed
    # print(f"Epoch {epoch} Loss: {epoch_loss / len(dataloader)}")

print("Training completed!")
