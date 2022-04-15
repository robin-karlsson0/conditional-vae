# Conditional VAE with partially observable data
Code for experimenting with Conditional VAEs and learning from partially observable data. The code is implemented in PyTorch Lightning.

The code implements a full-covariance VAE with minimalistic encoder $enc_{\phi}()$ and decoder $dec_{\theta}()$ models for approximating the posterior $q(z|x)$ and generative $p(x|z)$ distributions.

All data samples $x$ are corrupted by random masking, resulting partially observable data $\tilde{x}$. The VAE is optimized to discover a latent code $z$ which represents the uncorrupted data $x$ conditioned on the observable data $\tilde{x}$. The mapping from $z$ to $\hat{x}$ is optimized according to a partly uncorrupted $\tilde{x}^*$ data. The VAE is therefore required to learn the distribution of complete solutions from partial inputs and partial solutions.

![Screenshot from 2022-04-15 09-41-42](https://user-images.githubusercontent.com/34254153/163501932-f386b6ae-152a-436c-a994-e33ff3b0717c.png)

To run
```
python mnist_exp.py
```

Experiment parameters are specified from within `mnist_exp.py`.

The code generates Tensorboard plots and visualizations (test set samples, randomly sampled latent vectors $z$ from prior $p(z)$) easily accessible through vscode.

Plots
- Reconstruction $-log \: p(\hat{x})$
- KL-divergence $D_{KL}(q_{\phi}(z|x), p(z) )$
- neg. ELBO

References
- K. Murphy, Probabilistic Machine Learning: Advanced Topics, MIT Press, 2023
- D. Kingma, M. Welling, An Introduction to Variational Autoencoders, Foundations and Trends in Machine Learning, Vol. 12, 2019

Code largely borrowing from 
- W. Falcon, Variational Autoencoder Demystified With PyTorch Implementation, Towards Data Science, 2020