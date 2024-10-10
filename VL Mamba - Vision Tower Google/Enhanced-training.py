import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForImageClassification, MambaForCausalLM, AdamW, get_linear_schedule_with_warmup
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from timm.data.transforms_factory import create_transform
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

# Set the CUDA visible devices to 3, 4, and 5
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5"

# Specify the device (using cuda:0 which maps to cuda:3)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the vision model (MambaVision)
vision_model = AutoModelForImageClassification.from_pretrained("/scratch/faaraan/mamba-chat/nvidiaMambaVision-T-1K", trust_remote_code=True)
vision_model.eval().to(device)

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

# Create a MultiModal Connector with output_dim=2560 and move to device
mmc = MMC(input_dim=1000, output_dim=2560).to(device)  # Adjusted input_dim to 1000 for MambaVision

# Load the Mamba LLM and tokenizer, and move LLM to device
tokenizer = AutoTokenizer.from_pretrained("/scratch/faaraan/mamba-chat/mamba-2.8b-hf")
mamba_llm = MambaForCausalLM.from_pretrained("/scratch/faaraan/mamba-chat/mamba-2.8b-hf").to(device)

# Optionally, add new tokens to the tokenizer
# new_tokens = ["<new_token1>", "<new_token2>", ...]
# tokenizer.add_tokens(new_tokens)
# mamba_llm.resize_token_embeddings(len(tokenizer))
# torch.nn.init.normal_(mamba_llm.get_input_embeddings().weight[-len(new_tokens):], mean=0.0, std=0.02)

# Define a Dataset class to load image and Q&A pairs
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

        # Load image
        image = Image.open(image_path).convert("RGB")  # Ensure image is in RGB
        image_tensor = self.transform(image)

        return image_tensor, question, answer

# Define the image transformation
transform = create_transform(input_size=(3, 224, 224), is_training=False)

# Load the dataset
dataset = ImageQADataset("/scratch/faaraan/LLaVAData/data_llava_data_week_1_to_8.json", transform)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4)  # Adjust num_workers as needed

# Define optimizer to include mamba_llm and mmc parameters
optimizer = AdamW(list(mamba_llm.parameters()) + list(mmc.parameters()), lr=5e-5, weight_decay=0.01)

# Define scheduler
epochs = 64
total_steps = len(dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=int(0.1 * total_steps),  # 10% warm-up
    num_training_steps=total_steps
)

# Initialize GradScaler for mixed precision
scaler = GradScaler()

# Initialize TensorBoard writer
writer = SummaryWriter(log_dir="logs")

# Function to save model checkpoint
def save_checkpoint(epoch, mamba_llm, mmc, optimizer, scheduler, scaler, checkpoint_dir="checkpoints"):
    # Ensure the checkpoint directory exists
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    # Save the model, optimizer, scheduler, scaler, and epoch number
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
    torch.save({
        'epoch': epoch,
        'model_state_dict': mamba_llm.state_dict(),
        'mmc_state_dict': mmc.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
    }, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch}: {checkpoint_path}")

# Optional: Function to load checkpoint
def load_checkpoint(checkpoint_path, mamba_llm, mmc, optimizer, scheduler, scaler):
    checkpoint = torch.load(checkpoint_path)
    mamba_llm.load_state_dict(checkpoint['model_state_dict'])
    mmc.load_state_dict(checkpoint['mmc_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Checkpoint loaded from {checkpoint_path}, starting from epoch {epoch}")
    return epoch

# Training loop with checkpoint saving every 'save_every' epochs
def train_model(dataloader, mamba_llm, vision_model, mmc, tokenizer, optimizer, scheduler, scaler, writer, epochs=64, save_every=16, checkpoint_dir="checkpoints"):
    mamba_llm.train()
    mmc.train()  # Set MMC to training mode
    
    for epoch in range(1, epochs + 1):
        epoch_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}"):
            images, questions, answers = batch
            
            # Step 1: Process images through the vision model
            images = images.to(device)
            with torch.no_grad():
                visual_features = vision_model(images).logits  # Shape: [B, 1000]
            
            # Step 2: Pass visual features through the MMC
            visual_output = mmc(visual_features)  # Shape: [B, 2560]
            
            # Step 3: Tokenize text inputs (questions and answers) with increased max_length
            input_ids = tokenizer(
                questions,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512  # Adjust as needed
            ).input_ids.to(device)
            labels = tokenizer(
                answers,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512  # Adjust as needed
            ).input_ids.to(device)

            # Step 4: Embed visual features as a token and combine with text embeddings
            visual_token = visual_output.unsqueeze(1)  # Shape: [B, 1, 2560]
            text_embeddings = mamba_llm.get_input_embeddings()(input_ids)  # Shape: [B, N, 2560]
            combined_embeddings = torch.cat([visual_token, text_embeddings], dim=1)  # Shape: [B, N+1, 2560]

            # Step 5: Adjust labels to account for the visual token:
            max_input_length = combined_embeddings.size(1)  # N+1 (includes visual token)

            # Ensure the labels are truncated correctly
            labels_truncated = labels[:, :max_input_length - 1]  # Exclude visual token from label sequence length

            # Create a padded labels tensor with the right shape
            labels_padded = torch.full((labels.size(0), max_input_length), -100).to(device)  # Shape: [B, N+1]

            # Use the size of the truncated labels to correctly place them in the padded tensor
            num_labels_to_place = labels_truncated.size(1)  # Number of valid labels
            labels_padded[:, 1:num_labels_to_place + 1] = labels_truncated  # Place labels after visual token

            # Step 6: Forward pass through the LLM with combined embeddings and padded labels
            with autocast():
                outputs = mamba_llm(inputs_embeds=combined_embeddings, labels=labels_padded)
                loss = outputs.loss

            epoch_loss += loss.item()
            
            # Step 7: Backward pass and optimization
            scaler.scale(loss).backward()
            clip_grad_norm_(mamba_llm.parameters(), max_norm=1.0)  # Gradient clipping
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
        
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch}/{epochs}, Loss: {avg_epoch_loss:.4f}")
        
        # Log to TensorBoard
        writer.add_scalar("Loss/train", avg_epoch_loss, epoch)
        
        # Save checkpoint every 'save_every' epochs
        if epoch % save_every == 0:
            save_checkpoint(epoch, mamba_llm, mmc, optimizer, scheduler, scaler, checkpoint_dir)

# Optionally, load a checkpoint to resume training
# load_checkpoint('/scratch/faaraan/mamba-chat/checkpoints/checkpoint_epoch_45.pt', mamba_llm, mmc, optimizer, scheduler, scaler)

# Example of usage
train_model(
    dataloader, 
    mamba_llm, 
    vision_model, 
    mmc, 
    tokenizer, 
    optimizer, 
    scheduler, 
    scaler, 
    writer, 
    epochs=64,        # Total number of epochs
    save_every=16,    # Save model every 16 epochs
    checkpoint_dir="checkpoints"  # Directory to save checkpoints
)
