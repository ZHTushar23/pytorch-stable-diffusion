# import libraries
import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler
from torch.nn.parallel import DataParallel
from diffusion2 import Diffusion
from torch.optim import Adam,SGD, lr_scheduler
from v42_dataloader import NasaDataset  
from torch.utils.data import random_split, DataLoader
from cam import CAM

def get_time_embedding(timestep):
    # Shape: (160,)
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160) 
    # Shape: (1, 160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    # Shape: (1, 160 * 2)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)

# def get_time_embedding(timestep):
#     # Shape: (40,)
#     freqs = torch.pow(10000, -torch.arange(start=0, end=40, dtype=torch.float32) / 40) 
#     # Shape: (1, 40)
#     x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
#     # Shape: (1, 40 * 2)
#     return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)

#parameters
root_data_dir = "/nfs/rs/psanjay/users/ztushar1/multi-view-cot-retrieval/LES102_MultiView_100m_F2/"

batch_size = 8
Seq_Len    = 2    # 77
Dim        = 100  # 768
L          = 50   # Dim will be 100
n_epochs   = 200
train_losses = []
model_saved_path = "saved_model/diff1.pth"



device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
# initialize sampler
generator = torch.Generator(device=device)
generator.manual_seed(42)
sampler = DDPMSampler(generator)

# initialize model
model = Diffusion()
# model = CAM(in_channels=1,gate_channels=64)

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
        # X:(Batch Size, 2, height, width )
        # Y:(Batch Size, 2, height, width )
        # context: (Batch size, 2, 100)
        X_train, Y_train, context = data['cot'], data['reflectance'],data['angles']
        # Move tensor to the proper device
        X_train = X_train.to(device,dtype=torch.float)
        Y_train = Y_train.to(device,dtype=torch.float)
        context = context.to(device,dtype=torch.float)

        # print("X_train Shape: ",X_train.shape)
        # print("Y_train Shape: ",Y_train.shape)
        # print("Context Shape: ",context.shape)

        # sample timestep
        # timesteps = tqdm(sampler.timesteps)
        # print(len(sampler.timesteps))

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
        model_output = model(X_train_noisy, context, time_embedding)
        # model_output   = model(X_train_noisy)

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

# # Set up tqdm for the testing loop
# progress_bar_test = tqdm(total=len(test_loader), desc='Testing')

# # Testing loop
# model.eval()  # Set the model to evaluation mode
# with torch.no_grad():
#     for inputs, labels in test_loader:
#         # Forward pass for testing
#         outputs = model(inputs)
#         test_loss = criterion(outputs, labels)

#         # Update tqdm progress bar for testing
#         progress_bar_test.update(1)
#         progress_bar_test.set_postfix({'Test Loss': test_loss.item()})

# # Close the tqdm progress bar for testing
# progress_bar_test.close()

print(f'  Mean Test MSE loss:  {np.average(train_losses):.5f}' +f'  Std Test MSE loss: {np.std(train_losses):.5f} ')