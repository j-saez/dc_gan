# Pytorch DCGAN implementation

This is a pytorch implementation of DCGAN to get familiar with the pytorch framekwork and enjoy the process.

## Train DCGAN
At the moment DCGAN can only be trained with the MNIST dataset. To do so run:

```bash
    cd <root_to_dcgan>/dc_gan
    python train.py
'''

There are a few hyperparameters you can play with. Run the following command to get information about them:

```bash
    python train.py --help
'''

## Training notes

**D(x)** should return values close to 1 at the beginning of the training as x are real images.
**D(G(x))_1** is the output of the discriminator with fake images during the discriminator training. It should return values close to 0 at the beginning of the training as D(G(x)) are fake images.
**D(G(x))_2** is the output of the discriminator with fake images during the generator training. It should return values close to 0 at the beginning of the training as D(G(x)) are fake images.

As the training goes on, D(G(x)) should return values closer to 0.5.

It has been observed that when Batch normalization is not applied to the generator, nor to the discriminator, the above statements are not met. However, when batch normalization is applied to the generator and the discriminator, the model learns.

## Inference
**TODO**
