import os
if os.getcwd()[-6:] != 'dc_gan':
    message = 'Run the file from the the root:\n'
    message += 'cd dc_gan\n'
    message += 'python train.py'
    raise Exception(message)

############
## IMPORT ##
############

import time
import torch
import torch.nn as nn
from utils.params                   import Params
from utils.datasets                 import load_dataset, get_dataset_transforms
from torch.utils.data.dataloader    import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from models.discriminator           import Discriminator
from models.generator               import Generator
from torch.optim                    import Adam

###########################
## CONSTANTS AND GLOBALS ##
###########################

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {} to train the model.".format(DEVICE))

###########################
## Classes and functions ##
###########################

DATASETS_CHS = {'mnist': 1}

##########
## Main ##
##########

if __name__ == '__main__':

    print("DCGAN training loop")

    # Load params
    print("Loading params")
    hyperparms, dataparams = Params().get_params()
    print("\tParams loaded.")

    # Load the data
    print("Loading dataset")
    img_size = (dataparams.img_size,dataparams.img_size)
    transforms = get_dataset_transforms(img_size, DATASETS_CHS[dataparams.dataset_name])
    train_dataset = load_dataset(dataparams.dataset_name, transforms)
    train_dataloader = DataLoader(train_dataset,hyperparms.batch_size,shuffle=True)
    print("\tDataset loaded.")

    # Load the model
    print("Loading the models")
    (img_h, img_w) = img_size
    discriminator = Discriminator(DATASETS_CHS[dataparams.dataset_name],img_h,img_w).to(DEVICE)
    generator = Generator(hyperparms.z_dim, img_chs=DATASETS_CHS[dataparams.dataset_name]).to(DEVICE)
    print("\tModels loaded.")

    # Define optimizer and loss function
    print("Selecting optimizer and loss function")
    discriminator_optimizer = Adam(discriminator.parameters(), lr=hyperparms.lr, betas=(hyperparms.adam_beta1,hyperparms.adam_beta2))
    generator_optimizer = Adam(generator.parameters(), lr=hyperparms.lr, betas=(hyperparms.adam_beta1,hyperparms.adam_beta2))
    criterion = nn.BCELoss()
    print("\tDone.")

    total_train_baches = int(len(train_dataset) / hyperparms.batch_size)
    fixed_noise = torch.rand(hyperparms.batch_size, hyperparms.z_dim,1,1)

    # Training loop
    print('\n\nStart of the training process.\n')
    for epoch in range(hyperparms.total_epochs):
        epoch_init_time = time.perf_counter()
        for batch_idx, (real_imgs, _) in enumerate(train_dataloader):
            batch_init_time = time.perf_counter()

            # Data to device and to proper data type
            real_imgs = real_imgs.to(DEVICE).to(torch.float32)
            noise = torch.rand(hyperparms.batch_size,hyperparms.z_dim,1,1).to(DEVICE)
            fake_imgs = generator(noise)

            ## Train discriminator: max log(D(x)) + log(1-D(G(z)))
            real_disc_output = discriminator(real_imgs)
            loss_real_disc = criterion(real_disc_output, torch.ones_like(real_disc_output))
            fake_disc_output = discriminator(fake_imgs)
            loss_fake_disc = criterion(fake_disc_output, torch.zeros_like(real_disc_output))
            loss_disc = (loss_real_disc + loss_fake_disc) / 2.0

            discriminator.zero_grad()
            loss_disc.backward(retain_graph=True)
            discriminator_optimizer.step()

            ## Train generator: min log(1-D(G(z))) <--> max log(D(G(z)))
            fake_disc_output = discriminator(fake_imgs)
            loss_generator = criterion(fake_disc_output, torch.ones_like(fake_disc_output))

            generator.zero_grad()
            loss_generator.backward()
            generator_optimizer.step()

            batch_final_time = time.perf_counter()
            batch_exec_time = batch_final_time - batch_init_time
            
            if batch_idx % hyperparms.test_after_n_epochs == 0 and batch_idx !=0:
                # To be honest, in GANs the loss does not say much 
                print(f'Epoch {epoch}/{hyperparms.total_epochs} - Batch {batch_idx}/{total_train_baches} - Loss D {loss_disc:.6f} - Loss G {loss_generator:.6f} - Batch time {batch_exec_time:.6f} s.')

        epoch_final_time = time.perf_counter()
        epoch_exec_time = epoch_final_time - epoch_init_time
        if epoch % hyperparms.test_after_n_epochs == 0 and epoch !=0:
            # Test model
            with torch.no_grad():
                generator.eval()
                test_generated_img = generator(fixed_noise)
                generator.train()
