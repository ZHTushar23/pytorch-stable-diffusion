import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler

# WIDTH = 512
# HEIGHT = 512
# LATENTS_WIDTH = WIDTH // 8
# LATENTS_HEIGHT = HEIGHT // 8
Seq_Len, Dim   = 100,2
Batch_Size     = 2
WIDTH = 144
HEIGHT = 144
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8
# cfg = classifier free guidance, cfg weight=[1,14]
# n_inference_steps = no of steps to take during inference
# generate is the inference function
def generate(
    prompt,
    input_image=None,
    strength=0.8,
    sampler_name="ddpm",
    n_inference_steps=50,
    model=None,
    seed=None,
    device=None
):
    if not 0 < strength <= 1:
        raise ValueError("strength must be between 0 and 1")

    # Initialize random number generator according to the seed specified
    generator = torch.Generator(device=device)
    if seed is None:
        generator.seed()
    else:
        generator.manual_seed(seed)

    # (Batch_Size, Seq_Len, Dim)
    context = prompt
    if sampler_name == "ddpm":
        sampler = DDPMSampler(generator)
        sampler.set_inference_timesteps(n_inference_steps)
    else:
        raise ValueError("Unknown sampler value %s. ")

    # latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)
    input_image_shape = (Batch_Size,1,HEIGHT,WIDTH)

    model.to(device)
    input_image = torch.unsqueeze(input_image,0)
    context = torch.unsqueeze(context,0)
    input_image =    input_image.to(device,dtype=torch.float32)
    context     =     context.to(device,dtype=torch.float32)


    timesteps = tqdm(sampler.timesteps)
    for i, timestep in enumerate(timesteps):
        # (1, 320)
        time_embedding = get_time_embedding(timestep).to(device)

        # (Batch_Size, 4, Latents_Height, Latents_Width)
        # model_input = input_image

        # print(input_image.device,context.device,time_embedding.device)
        print("TimeStep: ",i)
        # model_output is the predicted noise
        # (Batch_Size, 1, Latents_Height, Latents_Width) -> (Batch_Size, 1, Latents_Height, Latents_Width)
        model_output = model(input_image, context, time_embedding)
        # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 4, Latents_Height, Latents_Width)
        input_image = sampler.step(timestep, input_image, model_output)

    # images = rescale(images, (-1, 1), (0, 2), clamp=True)
    # (Batch_Size, Channel, Height, Width) -> (Batch_Size, Height, Width, Channel)
    images = input_image.permute(0, 2, 3, 1)
    images = images.to("cpu").numpy()
    return images[0]
    
def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x

def get_time_embedding(timestep):
    # Shape: (160,)
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160) 
    # Shape: (1, 160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    # Shape: (1, 160 * 2)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)