from diffusers import StableDiffusionXLPipeline, DDPMScheduler
from transformers import CLIPTokenizer
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as Trans
from accelerate import Accelerator
from PIL import Image
import os

# Define the custom dataset class
class GreetingCardDataset(Dataset):
    def __init__(self, image_directory, prompts_file, transforms=None):
        self.image_directory = image_directory
        self.transforms = transforms
        with open(prompts_file, "r") as file:
            self.prompts = file.readlines()
            # Sort the images considering the format "image (1).jpg"
            self.image_files = sorted(
                os.listdir(image_directory),
                key=lambda x: int(x.split("(")[1].split(")")[0])
            )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_directory, self.image_files[index])
        img = Image.open(image_path).convert("RGB")
        if self.transforms:
            img = self.transforms(img)
        prompt = self.prompts[index].strip()
        return prompt, img

# Define transformations for the images
image_transforms = Trans.Compose([
    Trans.Resize((1024, 1024)),
    Trans.ToTensor(),
    Trans.Normalize([0.5]*3, [0.5]*3),
])

# Load dataset
dataset = GreetingCardDataset(
    image_directory=r"C:\Users\Jethro\PycharmProjects\OJT Project\dataset\images",
    prompts_file=r"C:\Users\Jethro\PycharmProjects\OJT Project\dataset\prompts.txt",
    transforms=image_transforms
)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Load the pre-trained Stable Diffusion XL model and pipeline
pipeline = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
pipeline.to("cuda")  # Move to GPU

# Prepare optimizer and scheduler
optimizer = torch.optim.AdamW(pipeline.unet.parameters(), lr=1e-5)
scheduler = DDPMScheduler(num_train_timesteps=1000)

# Initialize the accelerator
accelerator = Accelerator(mixed_precision="fp16")

# Tokenizer 
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

# Training loop
for epoch in range(5):
    for step, (text_prompt, image_data) in enumerate(dataloader):
        # Tokenize the text prompt
        text_inputs = tokenizer(text_prompt, return_tensors="pt", padding=True, truncation=True).input_ids.to("cuda")

        # Convert text inputs back to list of strings for the pipeline
        text_prompt_list = [tokenizer.decode(ids, skip_special_tokens=True) for ids in text_inputs]

        # Forward pass through the model
        with torch.cuda.amp.autocast():
            # Generate image based on prompts and input image
            output = pipeline(prompt=text_prompt_list, image=image_data)
            # Access the generated image
            generated_image = output.images[0]  # PIL Image

        # Convert PIL Image to tensor
        generated_image_tensor = Trans.ToTensor()(generated_image).unsqueeze(0).to("cuda")
        image_data = image_data.to("cuda")

        # Resize generated image tensor to match input image size
        if generated_image_tensor.size() != image_data.size():
            generated_image_tensor = Trans.Resize(image_data.size()[2:])(generated_image_tensor)

        # Clamp tensor values to [0, 1] before scaling
        generated_image_tensor = torch.clamp(generated_image_tensor, 0.0, 1.0)
        image_data = torch.clamp(image_data, 0.0, 1.0)

        # Ensure gradients are enabled for the model parameters
        for param in pipeline.unet.parameters():
            param.requires_grad = True

        # Ensure the tensors used for loss calculation require gradients
        generated_image_tensor.requires_grad = True
        image_data.requires_grad = True

        # Compute the loss between the generated image and the original image
        loss = torch.nn.functional.mse_loss(generated_image_tensor, image_data)

        # Backpropagation
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

        # Print progress
        if step % 10 == 0:
            print(f"Epoch {epoch}, Step {step}, Loss: {loss.item()}")

# Save the fine-tuned model
pipeline.save_pretrained(r"C:\Users\Jethro\PycharmProjects\OJT Project\ModelV2")

