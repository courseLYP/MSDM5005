import torch.nn as nn
import torch
import os
from torchvision.utils import save_image
import shutil


# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 128

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 128

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 200

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparameter for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

def denorm(x):
    # TANH [-1, 1]
    out = (x + 1) / 2
    return out.clamp(0, 1)

# Generator Code

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),

            nn.ConvTranspose2d( ngf*16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        output = self.main(input)
        return output
    
    
def inference(output_dir, state_dict_path, number_of_images = 2000):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)
    C = 3
    H = 128
    W = 128

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    generator = Generator(ngpu).to(device)
    generator.load_state_dict(torch.load(state_dict_path))
    generator.eval()
    with torch.inference_mode():
        for i in range(number_of_images):
            noise = torch.randn(1, 100, 1, 1, device=device)
            fake = generator(noise).detach().cpu()

            # reshape. C W H
            fake_image = fake.reshape(fake.size(0), C, W, H)
            fake_images_path = os.path.join(f"{output_dir}", f'fake_images_{i}.png')
            # print(f"{fake_images_path}, fake_image.shape={fake_image.shape}")
            save_image(denorm(fake_image), fake_images_path)