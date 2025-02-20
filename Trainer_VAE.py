
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import matplotlib.pyplot as plt

from Lpips import VGG16LPIPS
from Discriminator import Discriminator
from util import save_vae, load_vae
from torchmetrics.image.fid import FrechetInceptionDistance
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau


class trainer_VQ():
    def __init__(self, device, n_epochs, lr, model, training_vae_continue, diffuser=False, model_path=None):
        self.device = device
        self.n_epochs = n_epochs
        self.lr = lr
        self.training_continue = training_vae_continue
        self.perceptual_loss_fn = VGG16LPIPS().to(device)        
        self.model = model
        self.model.to(device)
        self.diffuser = diffuser
        self.optimizer = optim.AdamW(self.model.parameters(),lr=lr, eps=1e-6, amsgrad=True)
        self.scheduler_step = StepLR(self.optimizer, step_size=30, gamma=0.2)
        self.scheduler_plateau = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    
        if training_vae_continue:
            load_vae(self.model, self.optimizer, model_path)
        self.optimizer.param_groups[0]['lr'] = lr

        
    def train(self, train_dataloader, model_name, lpips_ratio=0.5):
        for epoch in range(self.n_epochs):
            train_loss = 0.0
            train_loss_tot = 0.0
            recon_loss_sum = 0.0
            per_loss_sum = 0.0
            scaler = torch.amp.GradScaler('cuda')
            self.model.to(self.device)
            self.model.train()
            for i, (batch, label) in enumerate(tqdm(train_dataloader)):
                # forward
                self.optimizer.zero_grad()
                
                x = batch

                x = x.to(self.device)
                with torch.amp.autocast('cuda'):
                    reconstructed, _, recon_loss  = self.model(x)

                    #perceptual loss
                    per_loss = self.perceptual_loss_fn(x, reconstructed)
                    
                    gen_base_loss = recon_loss + lpips_ratio * per_loss

                    total_gen_loss = gen_base_loss
               

                # Backprop for Generator
                
                scaler.scale(total_gen_loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                #total_gen_loss.backward()
                #self.optimizer.step()
                
                # 로그 저장
                train_loss += total_gen_loss.item()
                train_loss_tot += total_gen_loss.item()
                recon_loss_sum += recon_loss.item()
                per_loss_sum += per_loss.item()

                        # Epoch별 로그 출력
                if i%100==0:
                    print(f"===> Epoch: {epoch+1} "
                    f"Generator Loss: {train_loss/100} "
                    f"L2 Loss: {recon_loss_sum/100} "
                    f"Perceptual Loss: {per_loss_sum/100} "
                    f"Learning Rate: {self.optimizer.param_groups[0]['lr']}")
                    train_loss = 0.0
                    recon_loss_sum = 0.0
                    per_loss_sum = 0.0
            
            dataset_size = len(train_dataloader.dataset)

            self.scheduler_step.step()

            self.scheduler_plateau.step(train_loss_tot/dataset_size)
            
            
            save_vae(self.model, self.optimizer, model_name)
            if self.optimizer.param_groups[0]['lr']<1e-8:
                print("Finish Traning cause of LR limit")
                break

    def test(self, x):
        x = x.to(self.device)
        if self.diffuser:
            latents = self.model.encode(x)
            x_reconst = self.model.decode(latents)

        else:
            x_reconst, _, _ = self.model(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x_reconst = x_reconst.permute(0, 2, 3, 1).contiguous()


        fig, axes = plt.subplots(2, 4, figsize=(15, 6))  # 2행 num_images열

        # 원본 이미지 출력
        for i in range(4):
            axes[0, i].imshow(x.cpu()[i]*0.5+0.5)
            axes[0, i].set_title(f"Original {i+1}")
            axes[0, i].axis('off')

        # 재구성된 이미지 출력
        for i in range(4):
            axes[1, i].imshow(x_reconst.cpu().detach().numpy()[i]*0.5+0.5)
            axes[1, i].set_title(f"Reconstructed {i+1}")
            axes[1, i].axis('off')

        # 레이아웃 조정 및 출력
        plt.tight_layout()
        plt.show()


