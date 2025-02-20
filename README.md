# Classifier Free Guidance + V-prediction based Diffusion model

## Training
- DiT based model 사용
- Autoencoder는 VQ-VAE 사용
- Diffuser의 AutoencoderKL도 원한다면 불러와 사용 가능
- V-prediction based model 사용 (원한다면 eps, x prediction으로 사용 가능)
- loss는 SNR+1 Loss 사용 (SNR, Truncated, SNR+1 중 택 가능, PROGRESSIVE DISTILLATION FOR FAST SAMPLING OF DIFFUSION MODELS 논문 참조)
- ema 사용
- mixed precision training 사용

## Sampling
- DDIM 기반 Sampling 사용
- Classifier Free Guidance 사용 (식은 CFG 논문 기준이 아닌, stable diffusion에서 사용하는 식 사용)

## Result (Imagenet 중 9개 class 뽑아 학습, guidance scale = 3)
- <image src="https://github.com/user-attachments/assets/a884eb41-b4eb-414c-aa00-fadf1f6ed10e" width = "300"><image src="https://github.com/user-attachments/assets/5d55f206-d8ae-4270-b564-d31e545dceea" width = "300"><image src="https://github.com/user-attachments/assets/807676e4-3b5a-4cde-a18a-24be05cf8449" width = "300">
- <image src="https://github.com/user-attachments/assets/bdb06c0c-c01d-4e13-9471-4e4297ca2f03" width = "300"><image src="https://github.com/user-attachments/assets/888e29a1-7fb2-4150-81a0-9ed98d742b6f" width = "300"><image src="https://github.com/user-attachments/assets/1c252aaf-163d-4330-a22e-c842b3f9f580" width = "300">
- <image src="https://github.com/user-attachments/assets/e55e078b-ad58-4017-acc5-f5a830e60fcd" width = "300"><image src="https://github.com/user-attachments/assets/fb6d6443-ce56-48c1-9696-ea771135cfc6" width = "300"><image src="https://github.com/user-attachments/assets/314e6d47-1caf-4eec-8611-e29d23cab522" width = "300">
- <image src="https://github.com/user-attachments/assets/a7ecfc6b-471d-4061-a1b6-97a790c62417" width = "300"><image src="https://github.com/user-attachments/assets/9a9f8d91-2ffb-4a1c-b8f6-dab59319ad01" width = "300"><image src="https://github.com/user-attachments/assets/669a416b-570c-4322-8833-793da7dde6cf" width = "300">
- <image src="https://github.com/user-attachments/assets/3305febd-6558-4f99-9044-f16404cca70e" width = "300"><image src="https://github.com/user-attachments/assets/1fbf16fb-b1b2-4f93-b481-0cbe9862030c" width = "300"><image src="https://github.com/user-attachments/assets/47490aa2-9cde-4f9e-ab7c-7e8d65ffebf3" width = "300">
- <image src="https://github.com/user-attachments/assets/3160c556-831f-4ffc-bef8-ea176af924e4" width = "300"><image src="https://github.com/user-attachments/assets/a231b122-afb2-4be7-8602-cbd0b78c9341" width = "300"><image src="https://github.com/user-attachments/assets/780f447c-10b5-482f-809c-ef01d010a53c" width = "300">
- <image src="https://github.com/user-attachments/assets/c53daa91-7d47-4269-a44f-19bd66c174bb" width = "300"><image src="https://github.com/user-attachments/assets/861a6f44-3bf5-4e89-b679-020602766bf4" width = "300"><image src="https://github.com/user-attachments/assets/b6565154-d6f1-41a4-8f03-834eb1de43c7" width = "300">
- <image src="https://github.com/user-attachments/assets/81daca0f-beb2-41cf-b35a-2c04c865a6d9" width = "300"><image src="https://github.com/user-attachments/assets/55cba5a4-0e63-449b-9a05-6e3b33f11ad6" width = "300"><image src="https://github.com/user-attachments/assets/6adc5ef1-b8f9-41a3-b1b3-143e4db32afc" width = "300">
- <image src="https://github.com/user-attachments/assets/c61b4e68-41f3-48f4-adcc-527003744954" width = "300"><image src="https://github.com/user-attachments/assets/4f748c4e-1ae5-433c-8bc4-85e0ade62110" width = "300"><image src="https://github.com/user-attachments/assets/8afc296e-f281-4f15-bbd0-0af6e6f54271" width = "300">

