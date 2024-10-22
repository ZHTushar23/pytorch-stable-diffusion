import dm_pipeline
import torch
from v3_dataloader import NasaDataset 
from torch.utils.data import random_split, DataLoader
from diffusion_mini import DiffusionMini
DEVICE = "cpu"

ALLOW_CUDA = True
if torch.cuda.is_available() and ALLOW_CUDA:
    DEVICE = "cuda"
print(f"Using device: {DEVICE}")


# define context generator here
tt = 0
model_saved_path = "saved_model/miniDiff1.pth"
model = DiffusionMini(in_channels=1,interim_channels=32,out_channels=1)
model.load_state_dict(torch.load(model_saved_path))

# Create a separate generator for the random split
root_data_dir = "/nfs/rs/psanjay/users/ztushar1/multi-view-cot-retrieval/LES102_MultiView_100m_F2/"
split_generator = torch.Generator()
split_generator.manual_seed(13)  # You can choose any seed value

# Create an instance of your custom dataset
custom_dataset = NasaDataset(root_data_dir)
# Define the sizes for train, validation, and test sets
total_size = len(custom_dataset)
test_size = int(0.2 * total_size)
# Use random_split to split the dataset
_, test_data = random_split(
    custom_dataset, [total_size - test_size, test_size], generator=split_generator
)

test_loader = DataLoader(test_data, batch_size=1,shuffle=False)
## IMAGE TO IMAGE
data = test_loader.dataset[tt]
# get the data
X, input_image, context = data['reflectance'],data['cot'],data['angles']
print(input_image.shape, X.shape, context.shape)

# Higher values means more noise will be added to the input image, so the result will further from the input image.
# Lower values means less noise is added to the input image, so output will be closer to the input image.
strength = 0.9

## SAMPLER

sampler = "ddpm"
num_inference_steps = 50
seed = 42

output_image = dm_pipeline.generate(
    prompt=context,
    input_image=input_image,
    strength=strength,
    sampler_name=sampler,
    n_inference_steps=num_inference_steps,
    seed=seed,
    model=model,
    device=DEVICE
)
print(output_image.shape)
# Combine the input image and the output image into a single image.
np.save("diffMini.npy",output_image)