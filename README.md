# Pytorch DCGAN implementation

This is a pytorch implementation of [DCGAN](https://arxiv.org/abs/1511.06434) to get familiar with the pytorch framekwork and enjoy the process.

The code has been tested with pytorch 1.11.0 and torchvision 0.12.

## Train DCGAN
At the moment DCGAN can only be trained with the MNIST dataset. To do so run:

```bash
    cd <root_to_dcgan>/dc_gan
    python train.py
```

There are a few hyperparameters you can play with. Run the following command to get information about them:

```bash
    python train.py --help
```

### Training notes

1. **D(x)** should return values close to 1 at the beginning of the training as x are real images.
2. **D(G(x))_1** is the output of the discriminator with fake images during the discriminator training. It should return values close to 0 at the beginning of the training as **D(G(x))** are fake images.
3. **D(G(x))_2** is the output of the discriminator with fake images during the generator training. It should return values close to 0 at the beginning of the training as **D(G(x))** are fake images.
4. As the training goes on, **D(G(x))** should return values closer to 0.5.
5. It has been observed that when Batch normalization is not applied to the generator, nor to the discriminator, the above statements are not met. However, when batch normalization is applied to the generator and the discriminator, the model learns.

### Some results

**1. Lr=0.0002**
**2. epochs=200**
**3. z_dim=100**
**4. Batch norm applied to all the discriminator and generator layers but the last ones.**


#### Original
![Original](/images/original.jpeg)

#### Generated
![Generated](/images/generated.png)
