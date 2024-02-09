import torchvision
import matplotlib.pyplot as plt
import torch

from vae import VAE


def calculate_kl_loss(means, stds):
    return -1 * torch.sum(1.0 + 2.0 * stds - means.pow(2) - torch.exp(2.0 * stds), dim=1)


import os
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

"""
# Define the path to the Tiny ImageNet dataset on your machine
dataset_path = os.path.expanduser(
    os.path.join(
        '~',
        'Desktop',
        'tiny-imagenet-200'
    )
)

# Define transformations for the images
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize images to a consistent size
    transforms.ToTensor(),
])

# Use ImageFolder to load the dataset
tinyimagenet_train_dataset = ImageFolder(root=dataset_path + '/train', transform=transform)
tinyimagenet_test_dataset = ImageFolder(root=dataset_path + '/test', transform=transform)

# Create a DataLoader for the dataset
batch_size = 256
train_loader = DataLoader(tinyimagenet_train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(tinyimagenet_test_dataset, batch_size=batch_size, shuffle=False)

"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# use torchvision for mnist
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])
train_dataset = torchvision.datasets.MNIST(
    root='~/Desktop/mnist',
    train=True,
    transform=transform,
)
test_dataset = torchvision.datasets.MNIST(
    root='~/Desktop/mnist',
    train=False,
    transform=transform,
)

batch_size = 256

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
)

nr_epochs = 10

# Instantiate the model
input_channels = 1  # Assuming RGB images
latent_size = 20  # Adjust the latent size based on your requirements
nr_classes = 10
model = VAE(input_channels, latent_size, nr_classes)
model.train()
# adam optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
# cosine learning rate scheduler

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader) * nr_epochs)
mean_shape = None
model.to(device)
# Example of iterating through the dataset

load = False

if not load:
    for epoch in range(nr_epochs):
        epoch_loss = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            means, stds, x = model(images, labels)

            # keeping the abs if we switch to L1 loss
            reconstruction_loss = torch.abs(images - x)
            reconstruction_loss = reconstruction_loss.pow(2)
            reconstruction_loss = torch.sum(reconstruction_loss, dim=(1, 2, 3))
            kl_loss = calculate_kl_loss(means.view(x.size(0), -1), stds.view(x.size(0), -1))
            # beta_term = 4 * latent_size / (images.size(2) * images.size(3))
            beta_term = 0.5
            loss = torch.mean(reconstruction_loss + beta_term * kl_loss)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print(f'Epoch: {epoch + 1}, Loss: {loss.item():.3f}, Reconstruction: {torch.mean(reconstruction_loss).item():.3f}, KL: {torch.mean(kl_loss).item():.3f}')
            scheduler.step()
        epoch_loss /= len(train_loader)

        print(f'Epoch: {epoch + 1}, Loss: {epoch_loss:.3f}')

    # save model
    torch.save(model.state_dict(), 'model.pth')
else:
    model.load_state_dict(torch.load('model.pth'))

model.eval()

with torch.no_grad():
    initial_z = torch.randn(64, latent_size)
    initial_z = initial_z.to(device)
    class_info = torch.full((64,), 2)
    class_info = class_info.to(device)
    y = model.class_embedding(class_info)
    initial_z = initial_z + y
    x = model.dec_full_layer(initial_z)
    x = model.decoder_map(x)
    x = model.relu_activation(x)
    x = x.view(-1, *model.decoder_shape)
    x = model.decoder(x)
    x = x.cpu().detach()

# show images with pytorch
# reverse normalization
#x = x * 0.3081 + 0.1307
# reverse normalization
grid_img = torchvision.utils.make_grid(x, nrow=8)
plt.figure(figsize=(20, 20))
plt.imshow(grid_img.permute(1, 2, 0))
plt.tight_layout()
plt.show()
