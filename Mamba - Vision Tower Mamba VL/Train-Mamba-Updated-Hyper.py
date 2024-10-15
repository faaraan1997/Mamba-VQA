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
device = torch.device("cuda:6")
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

# Define the new MultiModal Connector with reduced output_dim
class MMC(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MMC, self).__init__()
        
        # Define layers for 2D selective scanning
        self.scan_layer_1 = torch.nn.Conv2d(in_channels=input_dim, out_channels=32, kernel_size=3, stride=1, padding=1) 
        self.scan_layer_2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        
        # Flatten layer before feeding into MLP
        self.flatten = torch.nn.Flatten()
        
        # MLP for further processing
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(64 * 28 * 28, output_dim),  # Adjust input_dim based on your vision model output
            torch.nn.GELU(),
            torch.nn.Linear(output_dim, output_dim)
        )
    
    def forward(self, visual_features):
        # Ensure the output shape is correct before passing to MLP
        batch_size = visual_features.size(0)  # Get the batch size
        print("Shape before MLP:", visual_features.shape)  # Debugging shape
        # If visual_features has more dimensions, you might need to handle that (e.g., flatten)
        visual_features = visual_features.view(batch_size, -1)  # Flatten if necessary
        print("Shape after Flattening:", visual_features.shape)  # Check shape after flatten
        return self.mlp(visual_features)  # MLP forward call
    
# Define image transformations with reduced size
transform = transforms.Compose([
    transforms.Resize((112, 112)),  # Resize to a smaller dimension
    transforms.ToTensor(),  # Convert to tensor
])

# Load the dataset
dataset = ImageQADataset("/scratch/faaraan/LLaVAData/data_llava_data_week_1_to_8.json", transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)  # Reduce batch size

# Initialize the MMC without height and width
mmc = MMC(input_dim=3, output_dim=1280).to(device)  # Reduce output_dim

# Load the Mamba LLM and tokenizer
tokenizer = AutoTokenizer.from_pretrained("/scratch/faaraan/mamba-chat/mamba-2.8b-hf")
mamba_llm = MambaForCausalLM.from_pretrained("/scratch/faaraan/mamba-chat/mamba-2.8b-hf").to(device)

# Freeze LLM parameters to prevent training
for param in mamba_llm.parameters():
    param.requires_grad = False

# Set the LLM to evaluation mode since it's frozen
mamba_llm.eval()

# Training configuration
epochs = 2  
optimizer = AdamW(mmc.parameters(), lr=1e-5)  # Adjust the optimizer and learning rate as needed
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)  # Define the loss function

# Create a directory to save the checkpoints
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# Mixed Precision Setup
scaler = torch.cuda.amp.GradScaler()

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
        visual_output = mmc(visual_features)  # Shape: [batch_size, 1280]

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
        visual_token = visual_output.unsqueeze(1)  # Shape: [batch_size, 1, 1280]

        # Step 5: Embed question tokens
        text_embeddings = mamba_llm.get_input_embeddings()(input_ids)  # Shape: [batch_size, N, 1280]

        # Step 6: Embed answer tokens
        answer_embeddings = mamba_llm.get_input_embeddings()(answer_input_ids)  # Shape: [batch_size, M, 1280]

        # Step 7: Combine embeddings
        combined_embeddings = torch.cat([visual_token, text_embeddings, answer_embeddings], dim=1)  # Shape: [batch_size, 1 + N + M, 1280]
        
        # Debugging shape of combined embeddings
        print(f"Combined Embeddings Shape: {combined_embeddings.shape}")

        # Step 8: Prepare attention mask
        answer_attention_mask = (answer_input_ids != tokenizer.pad_token_id).long()
        combined_attention_mask = torch.cat([ 
            torch.ones((attention_mask.size(0), 1), device=device),  # Visual token
            attention_mask,  # Question tokens
            answer_attention_mask  # Answer tokens
        ], dim=1)  # Shape: [batch_size, 1 + N + M]

        # Step 9: Prepare labels
        visual_ignore = torch.full((labels.size(0), 1), -100, dtype=labels.dtype, device=device)
        question_ignore = torch.full((labels.size(0), input_ids.size(1)), -100, dtype=labels.dtype, device=device)
        combined_labels = torch.cat([visual_ignore, question_ignore, labels], dim=1)  # Shape: [batch_size, 1 + N + M]

        # Mixed Precision Forward Pass
        with torch.cuda.amp.autocast():
            outputs = mamba_llm(
                inputs_embeds=combined_embeddings,
                attention_mask=combined_attention_mask,
                labels=combined_labels
            )
            loss = outputs.loss

        # Backpropagation
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()

    # Calculate and print the average loss for the epoch
    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch {epoch} Loss: {avg_loss:.4f}")

    # Save the model checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
    torch.save(mmc.state_dict(), checkpoint_path)

print("Training complete")