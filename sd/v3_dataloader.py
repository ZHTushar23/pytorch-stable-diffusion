'''
    Author: Zahid Hassan Tushar
    email: ztushar1@umbc.edu
'''


# Import Libraries
import os
import csv
from torch.utils.data import Dataset, random_split, DataLoader
import torch
import numpy as np
import torchvision.transforms as T
from angleEmbedding import get_embedder  
# torch.manual_seed(0)
SZA= torch.tensor([60., 40., 20.,  4.])
VZA= torch.tensor([ 60,  30,  15,   0, -15, -30, -60])

def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x.to(torch.float32)


def _read_csv(path: str):
    """Reads a csv file, and returns the content inside a list of dictionaries.
    Args:
      path: The path to the csv file.
    Returns:
      A list of dictionaries. Each row in the csv file will be a list entry. The
      dictionary is keyed by the column names.
    """
    with open(path, "r") as f:
        return list(csv.DictReader(f))

class NasaDataset(Dataset):
    def __init__(self, root_dir,L=50):
        self.root_dir = root_dir
        self.csv_file = _read_csv(root_dir+"data_files.csv")
        # self.filelist = os.listdir(root_dir)
        self.transform1 = T.Compose([T.ToTensor()])
        # self.transform2 = T.Compose([T.ToTensor()]) # define transformation
        self.embed, _ = get_embedder(L)
    
    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        reflectance_name = os.path.join(self.root_dir,self.csv_file[idx]['reflectance'])
        reflectance      = np.load(reflectance_name)[:,:,:1]
        # print(reflectance.shape)

        # cot profile
        cot_name         = os.path.join(self.root_dir,self.csv_file[idx]['cot'])


        # Angle Information
        sza = self.csv_file[idx]['SZA']
        vza = self.csv_file[idx]['VZA']

        sza_temp = torch.tensor([float(sza)])
        vza_temp = torch.tensor([float(vza)])

        # print(type(sza_temp),sza_temp)

        # batch size, 72,72,2
        cot_data         = np.load(cot_name)[:,:,:1]
        cot_data         = np.log(cot_data+1)
        # # Cloud Mask
        # cmask            = np.load(cot_name)[:,:,:1]
      
        # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
        sza_emb = self.embed(sza_temp)
        vza_emb = self.embed(vza_temp)
        context = torch.concat((sza_emb.unsqueeze(0),vza_emb.unsqueeze(0)),dim=0)

        # Convert to tensor
        reflectance = self.transform1(reflectance)
        cot_data    = self.transform1(cot_data)

        # # rescale
        cot_data = rescale(cot_data,(0, 6), (-1, 1))
        reflectance = rescale(reflectance,(0, 2.1272), (-1, 1))

        sample = {'reflectance': reflectance, 'cot': cot_data,'angles':context}
        return sample


if __name__=="__main__":
    # dataset_dir = "/nfs/rs/psanjay/users/ztushar1/COT_CER_Joint_Retrievals/one_thousand_profiles/Refl"
    dataset_dir = "/nfs/rs/psanjay/users/ztushar1/multi-view-cot-retrieval/LES102_MultiView_100m_F2/"

    train_data = NasaDataset(root_dir=dataset_dir)
    loader = DataLoader(train_data, batch_size=10,shuffle=False)
    print(len(loader.dataset))
    temp= []
    temp1= []
    for i in range(len(loader.dataset)):
        data = loader.dataset[i]
        # # get the data
        # X, Y, Z = data['reflectance'],data['cot'],data['angles']
        # print(Y.shape, X.shape, Z.shape)
        # print(type(Y))
        # print(Y.dtype)
        # break   
        x =    data['cot']
        temp.append(torch.max(x).item())
        temp1.append(torch.min(x).item())
    print(np.max(temp),np.min(temp1))