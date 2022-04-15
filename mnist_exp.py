import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from vanilla_vae import VanillaVAE

if __name__ == '__main__':

    # x: dim (28, 28) w. value range [0, 1]
    training_data = datasets.MNIST(
        root='data',
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_data = datasets.MNIST(
        root='data',
        train=False,
        download=True,
        transform=ToTensor()
    )

    train_dataloader = DataLoader(
        training_data,
        batch_size=128,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_dataloader = DataLoader(
        test_data,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    pl.seed_everything(1234)

    # Input channels
    in_ch = 1
    # Input size
    in_size = 32
    # Output channels
    out_ch = 1
    # Encoding vector dimension (from which latent vector is discovered)
    enc_dim = 256
    # Latent vector dimensions
    lat_dim = 8
    # Learning rate
    lr = 1e-3
    # List of filters per convolution layer (deconvolution is reversed)
    conv_chs = [8, 16, 32, 64]
    # Maximum and minimum random mask size ratios
    mask_p_max = 0.5
    mask_p_min = 0.25

    vae = VanillaVAE(
        in_ch,
        in_size,
        out_ch,
        enc_dim,
        lat_dim,
        lr,
        conv_chs,
        mask_p_max,
        mask_p_min,
    )
    trainer = pl.Trainer(gpus=1, max_epochs=100, progress_bar_refresh_rate=10)
    trainer.fit(vae, train_dataloader, val_dataloader)
