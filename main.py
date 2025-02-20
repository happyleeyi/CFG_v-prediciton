import torch
import numpy as np
from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision.transforms import Compose
import matplotlib.pyplot as plt

from Model import DiT
from Trainer import Trainer
from Trainer_VAE import trainer_VQ
from Model_VAE import VQVAE
from util import extract_features, VAE_diffuser

import kagglehub


# Download latest version
#path = kagglehub.dataset_download("anthonytherrien/dog-vs-cat")

#print("Path to dataset files:", path)

# Download latest version
#path = kagglehub.dataset_download("crawford/cat-dataset")

#print("Path to dataset files:", path)

# load dataset from the hub
# catface : ./cat_face/cat_face
# celeb : tonyassi/celebrity-1000      #25

dataset = load_dataset("benjamin-paine/imagenet-1k-256x256")
trial = 1000
model_base_name = './imagenet_dit_'
model_name = model_base_name + str(trial) + ".pt"
VAE_name = "./imagenet_VQVAE_8.pt"
feature_path = "./imagenet_feature.npz"
feature_trained = True
image_size = 256
latent_size = int(image_size/8)
embedding_dim = 4
channels = 3
batch_size = 8
batch_size_vae = 12
accum_batch = 64
training_steps = 1024       #training time step 수
num_steps = 256      #sampling time step 수
lr = 1e-4
lr_vae = 1e-5
epochs_vae = 300
epochs = 100
repeat_epoch = 5
num_classes = 100

use_diffuser_ae = True

training_vae_continue = True
testing_vae = False            #vae testing 할거면 1
training_state_vae = False     #training 해야되면 1, 모델있으면 0
z_channels = 4

training_state = 0       #training 단계면 1 sampling 단계면 0
save_folder = './'
num_images = 20
w = 3                    # guidance scale : 0 - uncond   1 - cond   1> - guidance
label = 99
gpu = 1                    #gpu 쓸지
device = "cuda" if torch.cuda.is_available() else "cpu"

dim = 1152
patch_size = 2
depth = 28
num_heads = 16
mlp_dim = 1152*4



vae_down_channels=[64, 128, 256, 512]
vae_mid_channels=[512, 512]
vae_down_sample=[True, True, True]
vae_attns = [False, False, False]
vae_num_down_layers = 2
vae_num_mid_layers = 2
vae_num_up_layers = 2
vae_z_channels = 4
vae_codebook_size = 8192
vae_norm_channels = 32
vae_num_heads = 4


transform = Compose([
            #transforms.CenterCrop(256),
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.Resize((image_size,image_size)),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t*2) - 1),
])

# define function 
class Custom_Dataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        image = example["image"]
        label = example["label"]

        if self.transform:
            image = self.transform(image)

        return image, label
    
def filter_by_label(example):
    return example["label"] in [1, 8, 24, 30, 37, 55, 71, 76, 99] # 원하는 라벨 조건

train_dataset = Custom_Dataset(dataset["train"].filter(filter_by_label), transform=transform)

# create dataloader
dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size_vae,
    shuffle=True
)

'''
dataiter = iter(dataloader)
plt.imshow(np.fliplr(np.rot90(np.transpose(next(dataiter)[0][0]/2+0.5), 3)))
plt.show()
'''

dit = DiT(latent_size, num_classes, dim, patch_size, depth, num_heads, mlp_dim, z_channels)


if use_diffuser_ae:
   vae = VAE_diffuser.from_pretrained(
    "stabilityai/sd-vae-ft-ema"
   )
   vae.to(device)
   training_vae_continue = False
   Trainer_vq = trainer_VQ(device, epochs_vae, lr_vae, vae, training_vae_continue, diffuser=True, model_path=VAE_name)
   if testing_vae:
      data_iter = iter(dataloader)
      batch = next(data_iter)  # 첫 번째 배치 가져오기

      images = batch[0]  # 이미지 데이터 추출

      # 첫 번째 이미지 선택
      first_image = images[0:4]  # 첫 번째 이미지
      print(first_image.shape)
      Trainer_vq.test(first_image)

else:
   vae = VQVAE(channels, vae_down_channels, vae_mid_channels, vae_down_sample, vae_attns, vae_num_down_layers, vae_num_mid_layers,
               vae_num_up_layers, vae_z_channels, vae_codebook_size, vae_norm_channels, vae_num_heads)


   Trainer_vq = trainer_VQ(device, epochs_vae, lr_vae, vae, training_vae_continue, diffuser=False, model_path=VAE_name)
   if training_state_vae:
      Trainer_vq.train(dataloader, VAE_name, 0.2)
   if testing_vae:
      data_iter = iter(dataloader)
      batch = next(data_iter)  # 첫 번째 배치 가져오기

      images = batch[0]  # 이미지 데이터 추출

      # 첫 번째 이미지 선택
      first_image = images[0:4]  # 첫 번째 이미지
      print(first_image.shape)
      Trainer_vq.test(first_image)

      


if feature_trained is False:
   extract_features(device, vae, dataloader, feature_path)


features_np = np.load(feature_path)
features_tensor = torch.from_numpy(features_np['features'])
labels_tensor = torch.from_numpy(features_np['labels'])
feature_dataset = TensorDataset(features_tensor, labels_tensor)
feature_dataloader = DataLoader(feature_dataset, batch_size=batch_size, shuffle=True)


if training_state:
   if trial == 0:
      model_name = None
   trainer = Trainer(device, feature_dataloader, latent_size, embedding_dim, batch_size, training_steps, num_steps, lr, model_name, vae, dit, mean_type='v')
   for epoch in range(epochs):
      print(str(trial+epoch*repeat_epoch+repeat_epoch)+" training start")
      if epoch == 0:
         load_ckpt = model_name
      else :
         load_ckpt = model_base_name + str(trial+epoch*repeat_epoch) + ".pt"
      save_ckpt = model_base_name + str(trial+epoch*repeat_epoch+repeat_epoch) + ".pt"
      
      trainer.training(load_ckpt, save_ckpt, repeat_epoch, accum_batch//batch_size)
      trainer.save_sample(1, label, w, save_file='./', diffuser=use_diffuser_ae)
      #trainer.fid_one_batch(trial+epoch+1, dt, dataloader)
else:
   trainer = Trainer(device, feature_dataloader, latent_size, embedding_dim, batch_size, training_steps, num_steps, lr, model_name, vae, dit, mean_type='v')
   trainer.save_sample(num_images, label, w, save_file=save_folder, diffuser=use_diffuser_ae)
   
      