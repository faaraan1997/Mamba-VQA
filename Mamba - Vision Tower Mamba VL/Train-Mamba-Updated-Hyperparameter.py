import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoTokenizer, AutoModelForImageClassification, MambaForCausalLM, AdamW
from tqdm import tqdm
from PIL import Image
import os

# Define the CUDA device
if torch.cuda.device_count() <= 7:
    raise ValueError(f"CUDA device 7 is not available. Only {torch.cuda.device_count()} CUDA devices found.")
device = torch.device("cuda:7")
print(f"Using device: {device}")

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
        question = item['conversations'][0]['value'].replace("<image>\n", "").strip()  # Clean the question
        answer = item['conversations'][1]['value'].strip()

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
vision_model = AutoModelForImageClassification.from_pretrained(
    "nvidia/MambaVision-T-1K",
    trust_remote_code=True
)
vision_model.eval().to(device)  # Move to specified device and set to eval mode

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
mmc = MMC(input_dim=1000, output_dim=2560).to(device)

# Load the Mamba LLM and tokenizer
tokenizer = AutoTokenizer.from_pretrained("/scratch/faaraan/mamba-chat/mamba-2.8b-hf")
mamba_llm = MambaForCausalLM.from_pretrained("/scratch/faaraan/mamba-chat/mamba-2.8b-hf").to(device)

# Freeze LLM parameters to prevent training
for param in mamba_llm.parameters():
    param.requires_grad = False

# Set the LLM to evaluation mode since it's frozen
mamba_llm.eval()

# Training configuration
epochs = 250
optimizer = AdamW(mmc.parameters(), lr=1e-5)  # Adjust the optimizer and learning rate as needed
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)  # Define the loss function

# Create a directory to save the checkpoints
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# Training loop
for epoch in range(1, epochs + 1):
    epoch_loss = 0
    mamba_llm.eval()  # Ensure LLM is in eval mode if it's not being trained
    for batch_idx, (images, questions, answers) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")):
        # Move images to specified device
        images = images.to(device)

        # Step 1: Extract visual features from the vision model
        with torch.no_grad():
            visual_output = vision_model(images)
            if isinstance(visual_output, dict):
                visual_features = visual_output.get('logits')  # Extract logits
            else:
                visual_features = visual_output

        # Step 2: Pass visual features through the MMC
        visual_output = mmc(visual_features)  # Shape: [batch_size, 2560]

        # Step 3: Tokenize questions and answers
        # Encode questions
        question_encoding = tokenizer(
            questions,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        input_ids = question_encoding.input_ids.to(device)
        attention_mask = question_encoding.attention_mask.to(device)

        # Encode answers (labels)
        with tokenizer.as_target_tokenizer():
            answer_encoding = tokenizer(
                answers,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
        # Move answer_input_ids to device before embedding
        answer_input_ids = answer_encoding.input_ids.to(device)
        labels = answer_input_ids.clone()
        # Replace padding token id's of the labels with -100 so it's ignored by the loss
        labels[labels == tokenizer.pad_token_id] = -100

        # Step 4: Embed visual features as a token
        visual_token = visual_output.unsqueeze(1)  # Shape: [batch_size, 1, 2560]

        # Step 5: Embed question tokens
        text_embeddings = mamba_llm.get_input_embeddings()(input_ids)  # Shape: [batch_size, N, 2560]

        # Step 6: Embed answer tokens
        answer_embeddings = mamba_llm.get_input_embeddings()(answer_input_ids)  # Shape: [batch_size, M, 2560]

        # Step 7: Combine embeddings
        # Concatenate visual token, question embeddings, and answer embeddings
        combined_embeddings = torch.cat([visual_token, text_embeddings, answer_embeddings], dim=1)  # Shape: [batch_size, 1 + N + M, 2560]

        # Step 8: Prepare attention mask
        # Create attention mask for visual token, question tokens, and answer tokens
        # Visual token: 1, question tokens: as per question_encoding, answer tokens: as per answer_encoding
        answer_attention_mask = (answer_input_ids != tokenizer.pad_token_id).long()
        combined_attention_mask = torch.cat([
            torch.ones((attention_mask.size(0), 1), device=device),  # Visual token
            attention_mask,  # Question tokens
            answer_attention_mask  # Answer tokens
        ], dim=1)  # Shape: [batch_size, 1 + N + M]

        # Step 9: Prepare labels
        # Labels should be [ignore for visual token, ignore for question tokens, answer tokens]
        # Create a tensor of -100 for visual token
        visual_ignore = torch.full((labels.size(0), 1), -100, dtype=labels.dtype, device=device)
        # Create a tensor of -100 for question tokens
        question_ignore = torch.full((labels.size(0), input_ids.size(1)), -100, dtype=labels.dtype, device=device)
        # Concatenate to form combined_labels
        combined_labels = torch.cat([
            visual_ignore,      # Ignore visual token
            question_ignore,    # Ignore question tokens
            labels               # Answer tokens
        ], dim=1)  # Shape: [batch_size, 1 + N + M]

        # Step 10: Forward pass through Mamba LLM
        outputs = mamba_llm(
            inputs_embeds=combined_embeddings,
            attention_mask=combined_attention_mask,
            labels=combined_labels
        )

        loss = outputs.loss  # Compute the loss

        # Accumulate loss
        epoch_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (optional but recommended)
        torch.nn.utils.clip_grad_norm_(mmc.parameters(), max_norm=1.0)

        optimizer.step()

    if epoch % 10 == 0 or epoch == epochs:
        checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch}.pt")
        torch.save({
            'epoch': epoch,
            'mmc_state_dict': mmc.state_dict(),
            'mamba_llm_state_dict': mamba_llm.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch}")

    # Print epoch loss
    print(f"Epoch {epoch} Loss: {epoch_loss / len(dataloader)}")

print("Training completed!")
