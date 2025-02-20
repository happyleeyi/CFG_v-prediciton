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
- ||||



