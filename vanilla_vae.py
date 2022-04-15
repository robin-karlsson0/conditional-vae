from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torch.distributions.multivariate_normal import MultivariateNormal


def get_random_mask(h, w, p_h_max, p_w_max, p_h_min=0.25, p_w_min=0.25):
    
    r_h = np.random.uniform(p_h_min, p_h_max)
    r_w = np.random.uniform(p_w_min, p_w_max)
    
    mask_h = int(np.floor(r_h * h))
    mask_w = int(np.floor(r_w * w))

    alpha_i = np.random.random()
    alpha_j = np.random.random()

    i = alpha_i * (h - mask_h)
    j = alpha_j * (w - mask_w)
    i = int(np.floor(i))
    j = int(np.floor(j))

    mask = np.zeros((h, w), dtype=bool)
    mask[i:i+mask_h, j:j+mask_w] = True

    return mask

def get_random_masks(h, w, p_h_max, p_w_max, p_h_min=0.25, p_w_min=0.25):
    mask_1 = get_random_mask(h, w, p_h_max, p_w_max, p_h_min, p_w_min)
    mask_2 = get_random_mask(h, w, p_w_max, p_w_max, p_h_min, p_w_min)
    # Remove common elements using compliment of mask_1
    mask_2 = np.logical_and(~mask_1, mask_2)

    return mask_1, mask_2

class Encoder(pl.LightningModule):
    '''
    '''
    def __init__(self, in_ch, enc_dim, lat_dim, in_size, conv_chs: List):
        super().__init__()

        self.in_ch = in_ch
        
        self.enc_dim = enc_dim
        self.lat_dim = lat_dim
        self.in_size = in_size
        self.conv_chs = conv_chs

        # Derived variables
        n_convs = len(self.conv_chs)
        self.feat_map_size = self.in_size // 2**n_convs
        self.feat_map_flatten_dim = self.conv_chs[-1] * self.feat_map_size**2

        #############################
        #  Encoding transformatios
        #############################
        layers = []
        conv_ch_prev = self.in_ch
        for conv_ch in self.conv_chs:
            layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        conv_ch_prev,
                        out_channels=conv_ch,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias=False),
                    nn.BatchNorm2d(conv_ch),
                    nn.LeakyReLU()
                    )
            )
            conv_ch_prev = conv_ch
        self.conv_encoder = nn.Sequential(*layers)

        self.fc_encoder = nn.Sequential(
            nn.Linear(self.feat_map_flatten_dim, self.enc_dim, bias=False),
            nn.BatchNorm1d(self.enc_dim),
            nn.LeakyReLU()
        )

        self.fc_mu = nn.Linear(self.enc_dim, self.lat_dim)
        self.fc_logvar = nn.Linear(self.enc_dim, self.lat_dim)
        self.fc_l_mat = nn.Linear(self.enc_dim, self.lat_dim**2)
    
    def forward(self, x):
        # Encoding vector
        feat_map = self.conv_encoder(x)
        enc_vec = torch.flatten(feat_map, start_dim=1)
        enc_vec = self.fc_encoder(enc_vec)
        # Distribution parameters
        mu = self.fc_mu(enc_vec)
        logvar = self.fc_logvar(enc_vec)
        l_mat = self.fc_l_mat(enc_vec)
        l_mat = l_mat.reshape(-1, self.lat_dim, self.lat_dim)  # (B, D, D)

        return mu, logvar, l_mat


class Decoder(pl.LightningModule):
    '''
    '''
    def __init__(self, out_ch, enc_dim, lat_dim, in_size, conv_chs: List):
        super().__init__()

        self.out_ch = out_ch
        self.enc_dim = enc_dim
        self.lat_dim = lat_dim
        self.in_size = in_size
        self.conv_chs = conv_chs

        # Derived variables
        n_convs = len(self.conv_chs)
        self.feat_map_size = self.in_size // 2**n_convs
        self.feat_map_flatten_dim = self.conv_chs[0] * self.feat_map_size**2

        ##############################
        #  Decoding transformations
        ##############################

        self.fc_decoder = nn.Sequential(
            nn.Linear(self.lat_dim, self.enc_dim, bias=False),
            nn.BatchNorm1d(self.enc_dim),
            nn.LeakyReLU(),
            nn.Linear(self.enc_dim, self.feat_map_flatten_dim, bias=False),
            nn.BatchNorm1d(self.feat_map_flatten_dim),
            nn.LeakyReLU()
        )

        layers = []
        for idx in range(len(self.conv_chs)-1):
            layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        self.conv_chs[idx],
                        out_channels=self.conv_chs[idx+1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                        bias=False),
                    nn.BatchNorm2d(self.conv_chs[idx+1]),
                    nn.LeakyReLU()
                )
            )
        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                        self.conv_chs[-1],
                        out_channels=self.conv_chs[-1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                        bias=False),
                nn.BatchNorm2d(self.conv_chs[idx+1]),
                nn.LeakyReLU(),
                nn.Conv2d(
                    self.conv_chs[-1],
                    out_channels=self.out_ch,
                    kernel_size=1),
                nn.Sigmoid()
                #nn.Tanh()
            )
        )
        self.conv_decoder = nn.Sequential(*layers)
        
    def forward(self, lat_vec):
        # Encoding vector
        enc_vec = self.fc_decoder(lat_vec)
        # Output map
        feat_map = enc_vec.reshape(
            -1, self.conv_chs[0], self.feat_map_size, self.feat_map_size)
        out = self.conv_decoder(feat_map)

        return out


class VanillaVAE(pl.LightningModule):
    '''
    Vanilla VAE with full-covariance Gaussian posterior.

    x -> enc(x) -> p(z|x) -> z -> dec(z) -> p(x|z) -> likelihood

    x: (Tensor) Input data (B, H, W)
    enc(): NN model estimating distribution parameters
    p(z|x): Posterior distribution for 'z'
    dec(): NN model estimating distribution parameters
    p(x|z): Likelihood of some code 'z' corresponding to some data 'x'

    Ref:
    - K. Murphy, Probabilistic Machine Learning: Advanced Topics, MIT Press,
          2023
    - D. Kingma, M. Welling, An Introduction to Variational Autoencoders, 
          Foundations and Trends in Machine Learning, Vol. 12, 2019
    - W. Falcon, Variational Autoencoder Demystified With PyTorch
          Implementation, Towards Data Science, 2020
    '''
    def __init__(
            self,
            in_ch,
            in_size,
            out_ch,
            enc_dim,
            lat_dim,
            lr,
            conv_chs: List,
            mask_p_max,
            mask_p_min,
            **kwargs) -> None:
        '''
        Args:
            in_ch: (int) Input map feature count
            in_size: (int) Input map spatial size
            out_ch: (int) Output map feature count
            enc_dim: (int) Data encoding dimension (from which latent code is
                           discovered)
            lat_dim: (int) Latent vector dimension
            lr: (float) Learning rate
            conv_chs: (list) Filter channels for conv models [4, 8, 16, 32]
        '''
        super().__init__()

        self.save_hyperparameters()

        self.lat_dim = lat_dim
        self.lr = lr

        self.mask_p_max = mask_p_max
        self.mask_p_min = mask_p_min

        # For the gaussian likelihood
        # self.log_scale = nn.Parameter(torch.Tensor([0.0]))

        self.encoder = Encoder(in_ch, enc_dim, lat_dim, in_size, conv_chs)

        rev_conv_chs = list(reversed(conv_chs))
        self.decoder = Decoder(out_ch, enc_dim, lat_dim, in_size, rev_conv_chs)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    def gaussian_likelihood(self, x_hat, logscale, x):
        '''
        Args:
            x_hat: (Tensor) Data generated from 'z' (B, 1, H, W)
            logscale: (Tensor) Constant 0 (1)
            x: (Tensor) Data which should be represented by 'z' (B, 1, H, W)
        '''
        scale = torch.exp(logscale)
        mean = x_hat
        dist = torch.distributions.Normal(mean, scale)

        # measure prob of seeing image under p(x|z)
        log_pxz = dist.log_prob(x)
        # Compute joint probability
        # p(x1, x2) = p(x1)*p(x2) == log_p(x1, x2) = log_p(x1) + log_p(x2)
        log_pxz = torch.sum(log_pxz, dim=(1, 2, 3))  
        return log_pxz
    
    @staticmethod
    def rec_mse(x_hat, x):
        '''
        Args:
            x_hat: (Tensor) Data generated from 'z' (B, 1 H, W)
            x: (Tensor) Data which should be represented by 'z' (B, 1, H, W)
        '''
        mse = torch.square(x_hat - x)
        mse = torch.mean(mse, dim=(1,2,3))
        return mse
    
    @staticmethod
    def rec_ce(x_rec, x, mask_2s, eps=1e-12):
        '''
        Args:
            x_rec: (Tensor) Data generated from 'z' (B, 1 H, W)
            x: (Tensor) Data which should be represented by 'z' (B, 1, H, W)
        '''
        ce = x * torch.log(x_rec + eps) + (1 - x) * torch.log(1 - x_rec + eps)
        mask_2s = torch.unsqueeze(mask_2s, dim=1)
        ce = ~mask_2s * ce
        ce = torch.sum(ce, dim=(1,2,3))
        return ce

    def kl_divergence(self, z, mu_post, l_mat_post):
        '''
        Computes D_KL value for Gaussian distributions q(z|x) and p(z) for
        given latent vector 'z' using Monte Carlo KL Divergence with abs()
        modification for conserving D_KL >= 0.

        Ref:
        - J. Liu, A Remedy For Negative Monte Carlo Estimated Values Of
              KL-divergence, 2021

        D_KL(p, q) = 1/N * sum( abs( log( p(z_i) / q(z_i) ) ) )

        Args:
            z: (Tensor) Latent vector (B, D)
            mu: (Tensor) Posterior distribution parameter vectors (B, D)
            std: (Tensor)

        Returns:
            (Tensor) Abs. D_KL values for each batch sample (B)
        '''
        device = z.get_device()

        # Single prior parametric distribution
        mu_prior = torch.zeros(self.lat_dim).to(device)
        cov_prior = torch.eye(self.lat_dim).to(device)
        distr_ps = MultivariateNormal(mu_prior, cov_prior)

        # Multiple posterior parametric distributions (for each batch sample)
        # NOTE Multivariate normal distribution w. diagonal covariance matrix
        distr_qs = MultivariateNormal(mu_post, scale_tril=l_mat_post, validate_args=False)

        # Element-wise log probability of 'z' given each distribution (B, D)
        log_pz = distr_ps.log_prob(z)  # (batch_n)
        log_qzx = distr_qs.log_prob(z)  # (batch_n)
        
        kl = torch.abs(log_qzx - log_pz)  # (batch_n)
        return kl
    
    def forward(self, x=None, z=None):

        if x is not None and z is None:
            mu, log_var, l_mat = self.encoder(x)
            l_mat = torch.tril(l_mat, diagonal=-1) + torch.diag_embed(torch.exp(log_var))
            q = MultivariateNormal(mu, scale_tril=l_mat, validate_args=False)
            z = q.rsample()
        elif x is None and z is None:
            mu = torch.zeros(self.lat_dim)
            l_mat = torch.eye(self.lat_dim)
            q = MultivariateNormal(mu, scale_tril=l_mat, validate_args=False)
            z = q.rsample()
        
        x_hat = self.decoder(z)

        return x_hat

    def training_step(self, batch, batch_idx):
        x, _ = batch

        device = x.get_device()

        # 28x28 --> 32x32
        x = torch.nn.functional.interpolate(x, size=32, mode='bilinear')

        x_obs = x.detach().clone()
        x_par = x.detach().clone()

        # Convert input to [-1, +1] range, with 0 equaling 'unknown'
        x_obs = 2*x_obs - 1

        # Random two-stage masking
        batch_n = x.shape[0]

        mask_2s = []
        for idx in range(batch_n):
            mask_1, mask_2 = get_random_masks(32, 32, self.mask_p_max, self.mask_p_max, self.mask_p_min, self.mask_p_min)
            full_mask = np.logical_or(mask_1, mask_2)

            full_mask = torch.tensor(full_mask, dtype=bool).to(device)
            mask_2 = torch.tensor(mask_2, dtype=bool).to(device)

            x_obs[idx][0][full_mask] = 0
            x_par[idx][0][mask_2] = 0
            mask_2s.append(mask_2)
        
        mask_2s = torch.stack(mask_2s)  # (B, H, W)

        # Encode x to get the mu and variance distribution parameters
        # mu: (B, D)
        # log_var: (B, D)
        # l_mat: (B, D, D)
        mu, log_var, l_mat = self.encoder(x_obs)  # x
        l_mat = torch.tril(l_mat, diagonal=-1) + torch.diag_embed(torch.exp(log_var))

        # Sample z from q(z|x)
        q = MultivariateNormal(mu, scale_tril=l_mat, validate_args=False)
        z = q.rsample()

        # Decoded data representation from latent code
        x_hat = self.decoder(z)  # (B, 1, H, W)

        # Reconstruction loss
        # rec_loss = self.gaussian_likelihood(x_hat, self.log_scale, x)
        # rec_loss = self.rec_mse(x_hat, x)
        rec_loss = -self.rec_ce(x_hat, x_par, mask_2s)  # x

        # D_KL( q(z|x), (p(z)) ) regularization term
        kl = self.kl_divergence(z, mu, l_mat)

        # Minimize negative ELBO
        # neg_elbo = kl - rec_loss
        neg_elbo = kl + rec_loss  # rec: CE
        neg_elbo = neg_elbo.mean()

        self.log_dict({
            'neg_elbo': neg_elbo,
            'kl': kl.mean(),
            'rec_loss': rec_loss.mean(),
        })

        return neg_elbo

    def validation_step(self, batch, batch_idx):

        x, y = batch

        device = x.get_device()

        # 28x28 --> 32x32
        x = torch.nn.functional.interpolate(x, size=32, mode='bilinear')

        x_obs = x.detach().clone()
        x_par = x.detach().clone()

        x_obs = 2*x_obs - 1

        # Random two-stage masking
        batch_n = x.shape[0]

        mask_2s = []
        for idx in range(batch_n):
            mask_1, mask_2 = get_random_masks(32, 32, self.mask_p_max, self.mask_p_max, self.mask_p_min, self.mask_p_min)
            full_mask = np.logical_or(mask_1, mask_2)

            full_mask = torch.tensor(full_mask, dtype=bool).to(device)
            mask_2 = torch.tensor(mask_2, dtype=bool).to(device)

            x_obs[idx][0][full_mask] = 0
            x_par[idx][0][mask_2] = 0.5
            mask_2s.append(mask_2)
        
        mask_2s = torch.stack(mask_2s)  # (B, H, W)

        x_hat = self(x_obs)  # x

        rec_loss = -self.rec_ce(x_hat, x_par, mask_2s)
        self.log_dict({'val_rec_loss': rec_loss})

        # Reconstruction visualizations
        if batch_idx == 0:
            x_par = x_par.cpu().numpy()
            x_obs = x_obs.cpu().numpy()
            x_hat = x_hat.cpu().numpy()

            batch_n = x.shape[0]
            batch_n = batch_n // 2
            plt.figure(figsize=((2*batch_n, 3*2)))
            for sample_idx in range(batch_n):
                plt.subplot(3, batch_n, sample_idx + 1 + 0*batch_n)
                plt.imshow(x_obs[sample_idx, 0], vmin=-1, vmax=1)
                plt.subplot(3, batch_n, sample_idx + 1 + 1*batch_n)
                plt.imshow(x_hat[sample_idx, 0], vmin=0, vmax=1)
                plt.subplot(3, batch_n, sample_idx + 1 + 2*batch_n)
                plt.imshow(x_par[sample_idx, 0], vmin=0, vmax=1)
                

            plt.tight_layout()
            self.logger.experiment.add_figure(
                f'viz_epoch{self.current_epoch:03d}', plt.gcf())
        
        # Sampled embedding visualizations
        if batch_idx == 1:
            device = x.get_device()
            mu = torch.zeros(self.lat_dim).to(device)
            l_mat = torch.eye(self.lat_dim).to(device)
            q = MultivariateNormal(mu, scale_tril=l_mat, validate_args=False)
            z_gen = q.sample_n(100)
            x_hat_gen = self.decoder(z_gen)
            self.logger.experiment.add_embedding(
                z_gen, label_img=x_hat_gen, global_step=self.current_epoch)
