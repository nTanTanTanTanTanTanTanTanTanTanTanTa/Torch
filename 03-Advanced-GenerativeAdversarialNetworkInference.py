import torch
import torch.nn as nn
from torchvision.utils import save_image

# Define the generator and discriminator architectures
latent_size = 64
hidden_size = 256
image_size = 784

G = nn.Sequential(
    nn.Linear(latent_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, image_size),
    nn.Tanh()
)

D = nn.Sequential(
    nn.Linear(image_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, 1),
    nn.Sigmoid()
)

# Load the generator and discriminator models
G.load_state_dict(torch.load('G.ckpt'))
G.eval()  # Set the generator to evaluation mode

D.load_state_dict(torch.load('D.ckpt'))
D.eval()  # Set the discriminator to evaluation mode

# Perform inference, generate images, or use the models for other tasks
# Example: Generate a fake image
z = torch.randn(1, latent_size)
fake_image = G(z)

# Save the generated image
fake_image = fake_image.reshape(1, 1, 28, 28)
save_image((fake_image + 1) / 2, 'generated_image.png')
