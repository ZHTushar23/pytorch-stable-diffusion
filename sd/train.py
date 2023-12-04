# import libraries
import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler
from dm_diffusion import Diffusion
from torch.optim import Adam,SGD, lr_scheduler
from dataloader import NasaDataset  
from torch.utils.data import random_split, DataLoader
from cam import CAM

# def get_time_embedding(timestep):
#     # Shape: (160,)
#     freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160) 
#     # Shape: (1, 160)
#     x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
#     # Shape: (1, 160 * 2)
#     return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)

def get_time_embedding(timestep):
    # Shape: (40,)
    freqs = torch.pow(10000, -torch.arange(start=0, end=40, dtype=torch.float32) / 40) 
    # Shape: (1, 40)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    # Shape: (1, 40 * 2)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)

#parameters
root_data_dir = "/nfs/rs/psanjay/users/ztushar1/pytorch-stable-diffusion/data/LES_vers1_multiangle_results/"
batch_size = 1
Seq_Len    = 2    # 77
Dim        = 100  # 768
L          = 50   # Dim will be 100
n_epochs   = 2 
train_losses = []
model_saved_path = "cam_dummy.pth"



device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
# initialize sampler
generator = torch.Generator(device=device)
generator.manual_seed(42)
sampler = DDPMSampler(generator)

# initialize model
# model = Diffusion()
model = CAM(in_channels=1,gate_channels=64)

# initialize dataloader
# Create an instance of your custom dataset
custom_dataset = NasaDataset(root_data_dir)

# Define the sizes for train, validation, and test sets
total_size = len(custom_dataset)
test_size = int(0.2 * total_size)
# Use random_split to split the dataset
train_data, test_data = random_split(
    custom_dataset, [total_size - test_size, test_size]
)

train_loader = DataLoader(train_data, batch_size=batch_size,shuffle=True)
# valid_loader = DataLoader(valid_data, batch_size=batch_size,shuffle=False)
test_loader = DataLoader(test_data, batch_size=batch_size,shuffle=False)

# specify optimizer functions
lr=0.0001
optimizer = Adam(model.parameters(), lr=lr,weight_decay=1e-05)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,threshold=1e-3,verbose=True)
criterion = torch.nn.MSELoss()
# start training loop

for epoch in range(1, n_epochs + 1):

    ###################
    # train the model #
    ###################
    model.to(device)
    model.train() # prep model for training
    for _, data in enumerate(train_loader, 1):
        # X:(Batch Size, 1, height, width )
        # Y:(Batch Size, nsza, nvza, 2, height, width )
        X_train, Y_train, context = data['cot'], data['reflectance'],data['angles']
        # Move tensor to the proper device
        X_train = X_train.to(device,dtype=torch.float)
        Y_train = Y_train.to(device,dtype=torch.float)
        context = context.to(device,dtype=torch.float)

        # # New Loop For angles
        # SZA= torch.tensor([60., 40., 20.,  4.])
        # VZA= torch.tensor([ 60,  30,  15,   0, -15, -30, -60])

        # for s in range(len(SZA)):
        #     for v in range(len(VZA)):
        #         Y_train  = Y[:,s,v,:,:]
        #         # Y_train = torch.unsqueeze(Y_train,1)
        #         # Move tensor to the proper device
        #         Y_train = Y_train.to(device,dtype=torch.float)

        #         # get context
        #         # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
        #         sza_emb,vza_emb = embed(SZA[s]),embed(VZA[v])
        #         context = torch.concat((sza_emb.unsqueeze(1),vza_emb.unsqueeze(1)),dim=1)

        # sample timestep
        timesteps = tqdm(sampler.timesteps)
        # print( type(timesteps))

        # get time embedding
        # (1, 320)
        time_embedding = get_time_embedding(sampler.timesteps[0]).to(device)
        # print(time_embedding.shape)
        
        # add noise
        X_train_noisy, Only_noise = sampler.add_noise(X_train, sampler.timesteps[0])

        # forward pass
        # model_output is the predicted noise
        # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 4, Latents_Height, Latents_Width)
        # model_output = model(X_train_noisy, context, time_embedding)
        model_output   = model(X_train_noisy)

        # compute loss
        loss = criterion(model_output,Only_noise)

        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # print(Y_train.shape)
        # print(output.shape)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # record training loss
        train_losses.append(loss.item())
    if (epoch + 1) % 5 == 0 or epoch == n_epochs - 1:
        torch.save(model.state_dict(), model_saved_path)

print(f'  Mean Test MSE loss:  {np.average(train_losses):.5f}' +f'  Std Test MSE loss: {np.std(train_losses):.5f} ')