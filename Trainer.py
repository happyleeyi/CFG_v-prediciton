import torch
import numpy as np
import matplotlib.pyplot as plt

from torchmetrics.image.fid import FrechetInceptionDistance

from torch.optim import AdamW

from tqdm.auto import tqdm

from diff import Diffusion
from util import save_checkpoint, load_checkpoint, EMA


class Trainer():
    def __init__(self, device, dataloader, image_size, channels, batch_size, training_steps, num_steps, lr, ckpt, vae, dit, mean_type='v'):
        self.device = device
        self.dataloader = dataloader
        self.image_size = image_size
        self.channels = channels
        self.batch_size = batch_size
        self.model = dit.to(self.device)
        self.training_steps = training_steps
        self.num_steps = num_steps
        self.lr = lr
        self.optimizer = AdamW(self.model.parameters(), lr=self.lr)
        self.ema = EMA(self.model, decay=0.999)
        if ckpt is not None:
            load_checkpoint(self.model, self.ema, self.optimizer, ckpt)
        self.optimizer.param_groups[0]['lr'] = lr
        self.diffusion = Diffusion(mean_type, training_steps)
        self.vae = vae

    def training(self, load_ckpt, save_ckpt, epochs, iter_to_accumulate=1, p_uncond = 0.1):
        self.model.to(self.device)

        scaler = torch.amp.GradScaler()

        self.model.train()

        for epoch in range(epochs):
            loss_sum=0
            self.optimizer.zero_grad()
            for step, (batch, label) in enumerate(tqdm(self.dataloader)):
                
                batch_size = batch.shape[0]
                batch = batch.to(self.device)
                label = label.to(self.device)

                # Algorithm 1 line 3: sample t uniformally for every example in the batch
                i = torch.randint(0, self.training_steps, (batch_size,), device=self.device).long()

                if torch.rand(1)<p_uncond:
                    label = None

                with torch.amp.autocast('cuda'):
                
                    loss = self.diffusion.p_losses(self.model, batch, label, i, loss_type="SNR+1")
                    loss = loss/iter_to_accumulate

                loss_sum+=loss.item()

                scaler.scale(loss).backward()

                if (step + 1) % iter_to_accumulate==0:

                    scaler.step(self.optimizer)
                    scaler.update()
                    #loss.backward()
                    #optimizer.step()
                    self.ema.update(self.model)
                    self.optimizer.zero_grad()
                if (step + 1) % 100 == 0:
                    print(f"===> Epoch: {epoch+1} "
                    f"Loss: {loss_sum/100} "
                    f"Learning Rate: {self.optimizer.param_groups[0]['lr']}")
                    loss_sum=0

                
                
        save_checkpoint(self.model, self.ema, self.optimizer, save_ckpt)
    def fid_one_batch(self, num, dt, real_dataloader):
        """
        ckpt_path: 확인하고자 하는 model의 checkpoint 경로.
                   내부에는 model, ema, optimizer state dict가 포함되어 있음.
        """
        # 1) 모델 상태 동기화 (체크포인트 로드)
        self.model.eval()

        # 2) Dataloader에서 단일 batch 추출
        #    여러 batch 평균이 아닌, 여기서는 예시로 첫 batch만 사용
        batch = next(iter(real_dataloader))
        
        real_images = batch["pixel_values"].to(self.device)[:10]
        real_images_01 = (real_images / 2) + 0.5

        # 3) 모델에서 생성된 이미지를 얻어오기 (Diffusion reverse + VAE decode 등)
        #    여기서는 간단 예시로 batch_size만큼 샘플링
        fake = []
        with torch.no_grad():
            # Diffusion Reverse 단계를 수행해서 fake latent를 얻는다 가정
            fake = self.save_sample(dt, real_images_01.shape[0], './', True, True)

            fake_images_01 = torch.tensor(np.array(fake)).to(self.device)/2+0.5

        # 4) torchmetrics의 FrechetInceptionDistance 사용
        #    normalize=True => 입력이 [0,1] 범위라면 내부적으로 [-1,1]로 바꿔 Inception 입력에 맞춤
        fid_metric = FrechetInceptionDistance(normalize=True).to(self.device)

        # real, fake 이미지 업데이트
        fid_metric.update(real_images_01, real=True)
        fid_metric.update(fake_images_01, real=False)

        # 5) FID 계산
        fid_score = fid_metric.compute().item()

        fid_log_file = "fid_scores.txt"
        with open(fid_log_file, "a") as f:
            f.write(f"{num} FID Score: {fid_score}\n")

        return fid_score
    def sampling(self, n_images, label, w, store_each_step=False):
        self.ema.apply_shadow()
        self.ema.model.to(self.device)
        self.ema.model.eval()
        with torch.no_grad():
            samples = self.diffusion.sample_images(self.ema.model, n_images, self.num_steps, self.channels, self.image_size, label, w, store_each_step)
        self.ema.restore()
        return samples
    def save_sample(self, n_images, label, w, save_file=None, diffuser=False):
        # 1) 샘플링 함수에서 얻어온 latent들
        #    리턴 shape => (n_images, 1, C, H, W) 형태 (numpy 배열 혹은 리스트)
        samples = self.sampling(n_images, label, w, store_each_step=False)

        # 2) numpy -> torch 변환 및 shape 정리
        samples_torch = torch.tensor(samples, device=self.device)

        samples_torch = samples_torch.squeeze(1)                   # (n_images, C, H, W)
        samples_torch = samples_torch.squeeze(1)

        # 3) VAE의 quantize & decode
        #    - VQ 모델인 경우, 'quantize'로 latent space 정제 후, 'decode'로 이미지 복원
        #    - 만약 단순 AutoEncoder라면 quantize 과정 없이 decode만 할 수도 있음
        if diffuser:
            decoded = self.vae.decode(samples_torch)            # (n_images, C, H, W)
        else:
            latents, _ = self.vae.quantize(samples_torch)  # shape => (n_images, C, H, W) 유지
            decoded = self.vae.decode(latents)            # (n_images, C, H, W)

        # 4) 원래 [-1, 1] 범위라면  (x/2 + 0.5)로 [0, 1] 범위로 맞춤
        decoded = decoded / 2 + 0.5

        # 6) 특정 이미지를 파일로 저장하기 (예: 첫 번째 이미지를 시각화)
        if save_file is not None:
            # decoded.shape = (n_images, C, H, W)
            # 첫 번째 이미지를 가져와서 (C,H,W)->(H,W,C) 변환 후 plt로 저장
            for i in range(n_images):
                plt.imshow(decoded[i].permute(1, 2, 0).cpu().detach().numpy())
                plt.savefig(save_file+str(i))

        # 7) 최종적으로 (n_images, C, H, W) 이미지를 numpy로 리턴
        return decoded.cpu().detach().numpy()



