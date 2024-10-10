import os
import json
import torch
from transformers import AutoTokenizer, AutoModel, MambaForCausalLM, AdamW
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from timm.data.transforms_factory import create_transform
from tqdm import tqdm

# Set the CUDA visible devices to 3, 4, and 5
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5"

# Specify the device (first visible device becomes device 0)
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

# Load the vision model (ViT) and move to device
vision_model = AutoModel.from_pretrained("/scratch/faaraan/mamba-chat/googlevit-base-patch16-224")
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
mmc = MMC(input_dim=768, output_dim=2560).to(device)

# Load the Mamba LLM and tokenizer, and move LLM to device
tokenizer = AutoTokenizer.from_pretrained("/scratch/faaraan/mamba-chat/mamba-2.8b-hf")
mamba_llm = MambaForCausalLM.from_pretrained("/scratch/faaraan/mamba-chat/mamba-2.8b-hf").to(device)

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
        image = Image.open(image_path)
        image_tensor = self.transform(image)

        return image_tensor, question, answer

# Define the image transformation
transform = create_transform(input_size=(3, 224, 224), is_training=False)

# Load the dataset
dataset = ImageQADataset("/scratch/faaraan/LLaVAData/data_llava_data_week_1_to_8.json", transform)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Define optimizer
optimizer = AdamW(mamba_llm.parameters(), lr=5e-5)

# Function to save model checkpoint
def save_checkpoint(epoch, mamba_llm, optimizer, checkpoint_dir="checkpoints"):
    # Ensure the checkpoint directory exists
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    # Save the model, optimizer, and epoch number
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
    torch.save({
        'epoch': epoch,
        'model_state_dict': mamba_llm.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch}: {checkpoint_path}")

# Training loop with checkpoint saving every 15 epochs
def train_model(dataloader, mamba_llm, vision_model, mmc, tokenizer, optimizer, epochs=64, save_every=16, checkpoint_dir="checkpoints"):
    mamba_llm.train()
    
    for epoch in range(1, epochs + 1):
        epoch_loss = 0
        for batch in tqdm(dataloader):
            images, questions, answers = batch
            
            # Step 1: Process images through the vision model
            images = images.to(device)
            with torch.no_grad():
                visual_features = vision_model(images).last_hidden_state  # Shape: [B, 196, 768]
            visual_output = mmc(visual_features.mean(dim=1))  # Shape: [B, 2560]
            # Step 2: Tokenize text inputs (questions and answers)
            input_ids = tokenizer(questions, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
            labels = tokenizer(answers, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)

            # Debugging: Check dimensions of tokenized inputs and labels
            print(f"Input IDs shape: {input_ids.shape}, Labels shape: {labels.shape}")

            # Step 3: Embed visual features as a token and combine with text embeddings
            visual_token = visual_output.unsqueeze(1)  # Shape: [B, 1, 2560]
            text_embeddings = mamba_llm.get_input_embeddings()(input_ids)
            combined_embeddings = torch.cat([visual_token, text_embeddings], dim=1)  # Shape: [B, N+1, 2560]

            # Debugging: Check dimensions of combined embeddings
            print(f"Combined embeddings shape: {combined_embeddings.shape}")

            # Step 4: Adjust labels to account for the visual token:
            # Truncate or pad labels to match the input length (N+1)
            max_input_length = combined_embeddings.size(1)  # This is N+1, including the visual token

            # Ensure the labels are truncated correctly
            labels_truncated = labels[:, :max_input_length - 1]  # Truncate labels to match the input sequence length (excluding visual token)

            # Create a padded labels tensor with the right shape
            labels_padded = torch.full((labels.size(0), max_input_length), -100).to(device)  # Shape: [B, N+1]

            # Use the size of the truncated labels to correctly place them in the padded tensor
            num_labels_to_place = labels_truncated.size(1)  # This is how many actual labels we have after truncation

            # Make sure we only assign as many labels as we have to avoid size mismatch
            labels_padded[:, 1:num_labels_to_place + 1] = labels_truncated  # Place labels after the visual token

            # Debugging: Check dimensions of padded labels
            print(f"Labels padded shape: {labels_padded.shape}, Truncated labels shape: {labels_truncated.shape}")

            # Step 5: Forward pass through the LLM with combined embeddings and padded labels
            outputs = mamba_llm(inputs_embeds=combined_embeddings, labels=labels_padded)
            loss = outputs.loss
            epoch_loss += loss.item()

            # Step 6: Backward pass and optimization
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(f"Epoch {epoch}/{epochs}, Loss: {epoch_loss/len(dataloader)}")
        
        # Save checkpoint every 'save_every' epochs
        if epoch % save_every == 0:
            save_checkpoint(epoch, mamba_llm, optimizer, checkpoint_dir)

# Example of usage
train_model(
    dataloader, 
    mamba_llm, 
    vision_model, 
    mmc, 
    tokenizer, 
    optimizer, 
    epochs=64,        # Total number of epochs
    save_every=15,    # Save model every 15 epochs
    checkpoint_dir="checkpoints"  # Directory to save checkpoints
)
