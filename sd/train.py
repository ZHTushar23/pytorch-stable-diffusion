# import libraries
import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler
from dm_diffusion import Diffusion
from torch.optim import Adam,SGD, lr_scheduler

device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
# initialize sampler
generator = torch.Generator(device=device)
generator.manual_seed(42)
sampler = DDPMSampler(generator)

# initialize model
model = Diffusion()

# initialize dataloader


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
        X_train, Y_train, angles = data['cot'], data['reflectance']
        
        # Y_train = torch.unsqueeze(Y_train,1)
        # Move tensor to the proper device
        X_train = X_train.to(device,dtype=torch.float)
        Y_train = Y_train.to(device,dtype=torch.float)

        # get context
        context = AngleEmbedding(angles)

        # sample timestep
        timesteps = tqdm(sampler.timesteps)

        # get time embedding
        # (1, 320)
        time_embedding = get_time_embedding(timesteps).to(device)
        
        # add noise
        X_train_noisy, Only_noise = sampler.add_noise(X_train, sampler.timesteps[0])
        
        # forward pass
        # model_output is the predicted noise
        # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 4, Latents_Height, Latents_Width)
        model_output = model(X_train_noisy, context, time_embedding)

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