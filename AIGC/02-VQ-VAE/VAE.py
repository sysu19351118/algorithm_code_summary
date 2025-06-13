import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau

class ResidualQuantizer(nn.Module):
    def __init__(self, num_quantizers, codebook_size, latent_dim, commitment_cost=0.25):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.latent_dim = latent_dim
        self.commitment_cost = commitment_cost
        
        # Initialize multiple codebooks
        self.codebooks = nn.ModuleList([
            nn.Embedding(codebook_size, latent_dim) 
            for _ in range(num_quantizers)
        ])
        
        # Initialize codebooks with uniform distribution
        for codebook in self.codebooks:
            codebook.weight.data.uniform_(-1.0/codebook_size, 1.0/codebook_size)
    
    def forward(self, z):
        # z shape: [batch_size, latent_dim]
        batch_size = z.shape[0]
        
        quantized = 0
        residual = z
        all_indices = []
        all_losses = []
        
        for i in range(self.num_quantizers):
            # Calculate distances to codebook vectors
            distances = (torch.sum(residual**2, dim=1, keepdim=True) 
                        - 2 * torch.matmul(residual, self.codebooks[i].weight.t()))
            
            # Get closest codebook indices
            indices = torch.argmin(distances, dim=1)
            quantized_i = self.codebooks[i](indices)
            
            # Update residual
            residual = residual - quantized_i.detach()
            quantized = quantized + quantized_i
            
            # Collect indices and losses
            all_indices.append(indices)
            
            # Commitment loss
            e_latent_loss = F.mse_loss(quantized_i.detach(), z)
            q_latent_loss = F.mse_loss(quantized_i, z.detach())
            loss = q_latent_loss + self.commitment_cost * e_latent_loss
            all_losses.append(loss)
        
        # Straight-through estimator for gradient
        quantized = z + (quantized - z).detach()
        
        # Average the losses
        quant_loss = torch.stack(all_losses).mean()
        
        # Stack all indices
        all_indices = torch.stack(all_indices, dim=1)  # [batch_size, num_quantizers]
        
        return quantized, quant_loss, all_indices

class MNISTRQVAE(pl.LightningModule):
    def __init__(self, latent_dim=32, num_quantizers=4, codebook_size=128, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.latent_dim = latent_dim
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)  # Output continuous latent representation
        )
        
        # Residual Quantizer
        self.quantizer = ResidualQuantizer(
            num_quantizers=num_quantizers,
            codebook_size=codebook_size,
            latent_dim=latent_dim
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Tanh()
        )
        
        self.rec_loss = nn.MSELoss(reduction='sum')
    
    def forward(self, x):
        # Flatten input
        x_flat = x.view(-1, 784)
        
        # Encode
        z = self.encoder(x_flat)
        
        # Quantize
        z_q, quant_loss, indices = self.quantizer(z)
        
        # Decode
        x_recon = self.decoder(z_q)
        
        return x_recon, z, z_q, quant_loss, indices
    
    def sample(self, num_samples):
        # Sample random indices for each quantizer
        indices = torch.randint(0, self.codebook_size, 
                              (num_samples, self.num_quantizers),
                              device=self.device)
        
        # Sum the codebook vectors
        z_q = 0
        for i in range(self.num_quantizers):
            z_q = z_q + self.quantizer.codebooks[i](indices[:, i])
        
        # Decode
        return self.decoder(z_q).view(-1, 1, 28, 28)
    
    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_recon, z, z_q, quant_loss, _ = self(x)
        
        # Reconstruction loss
        recon_loss = self.rec_loss(x.view(-1, 784), x_recon)
        
        # KL divergence (optional, can be removed for pure quantization)
        kl_loss = -0.5 * (1 + torch.log(torch.ones_like(z)) - z.pow(2) - torch.ones_like(z)).sum()
        
        # Total loss
        total_loss = recon_loss + quant_loss + 0.1 * kl_loss  # Adjust weights as needed
        
        # Logging
        self.log("recon_loss", recon_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("quant_loss", quant_loss, on_step=True, on_epoch=True)
        self.log("kl_loss", kl_loss, on_step=True, on_epoch=True)
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_recon, _, _, quant_loss, _ = self(x)
        
        recon_loss = self.rec_loss(x.view(-1, 784), x_recon)
        total_loss = recon_loss + quant_loss
        
        self.log("val_recon_loss", recon_loss, on_step=False, on_epoch=True)
        self.log("val_quant_loss", quant_loss, on_step=False, on_epoch=True)
        self.log('val_loss', total_loss, on_step=False, on_epoch=True)
        
        return total_loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True),
            'monitor': 'val_loss',
            'interval': 'epoch',
            'frequency': 1
        }
        return [optimizer], [scheduler]

# if __name__ == "__main__":
#     model = MNISTVAE(64)
    