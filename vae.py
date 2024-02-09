import torch
import torch.nn as nn


class VAE(nn.Module):

    def __init__(self, input_channels, latent_size, nr_classes, dropout=0.3):

        super(VAE, self).__init__()
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.latent_size = latent_size
        # embedder for the class information
        self.class_embedding = nn.Embedding(nr_classes, latent_size)
        self.encoder_dim = 256 * 4 * 4
        self.enc_full_layer = nn.Linear(self.encoder_dim, 256)
        self.dec_full_layer = nn.Linear(latent_size, 256)
        self.latent_head = nn.Linear(256, latent_size * 2)

        self.decoder_map = nn.Linear(256, self.encoder_dim)
        self.decoder_shape = (256, 4, 4)
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.ConvTranspose2d(64, input_channels, kernel_size=4, stride=2, padding=1),
        )

        self.relu_activation = nn.ReLU()

    def forward(self, x, y):

        for enc_layer in self.encoder:
            x = enc_layer(x)

        x = x.view(x.size(0), -1)
        x = self.enc_full_layer(x)
        x = self.latent_head(x)
        means, log_var = torch.split(x, self.latent_size, dim=1)
        stds = log_var * 0.5

        noise = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        noise_value = noise.sample(sample_shape=stds.shape)
        noise_value = noise_value.squeeze(2)
        noise_value = noise_value.to(x.device)
        latent_value = means + noise_value * torch.exp(stds)
        embedded_classes = self.class_embedding(y)
        latent_value = latent_value + embedded_classes
        latent_value = self.dec_full_layer(latent_value)
        latent_value = self.decoder_map(latent_value)
        latent_value = self.relu_activation(latent_value)
        latent_value = latent_value.view(-1, *self.decoder_shape)
        x = self.decoder(latent_value)

        return means, stds, x
