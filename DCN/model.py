import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.BatchNorm1d(in_features),
            nn.ReLU(),
            nn.Linear(in_features, in_features),
            nn.BatchNorm1d(in_features)
        )
        
    def forward(self, x):
        return F.relu(x + self.block(x))

class DeepClusteringNetwork(nn.Module):
    def __init__(self, input_dim=12, hidden_dims=[128, 64, 32], latent_dim=16, n_clusters=3):
        super(DeepClusteringNetwork, self).__init__()
        
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.BatchNorm1d(hidden_dims[2]),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims[2], latent_dim)
        )
        
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[2]),
            nn.BatchNorm1d(hidden_dims[2]),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims[2], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims[1], hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims[0], input_dim)
        )
        
        # Clustering layer
        self.clustering_layer = nn.Linear(latent_dim, n_clusters)
        
    def forward(self, x):
        # Encoder
        z = self.encoder(x)
        
        # Decoder
        x_recon = self.decoder(z)
        
        # Clustering
        q = self.clustering_layer(z)
        q = F.softmax(q, dim=1)
        
        return x_recon, z, q

class DCNLoss(nn.Module):
    def __init__(self, lambda_recon=1.0, lambda_cluster=1.0):
        super().__init__()
        self.lambda_recon = lambda_recon
        self.lambda_cluster = lambda_cluster

    def forward(self, x, x_recon, q):
        # Reconstruction loss
        recon_loss = F.mse_loss(x_recon, x)
        
        # Clustering loss (KL divergence) with stabilization
        p = self.target_distribution(q)
        epsilon = 1e-6
        cluster_loss = F.kl_div((q + epsilon).log(), p, reduction='batchmean')
        
        # Total loss
        total_loss = self.lambda_recon * recon_loss + self.lambda_cluster * cluster_loss
        return total_loss, recon_loss, cluster_loss

    @staticmethod
    def target_distribution(q):
        p = q ** 2 / q.sum(0)
        p = p / p.sum(1, keepdim=True)
        return p.detach()