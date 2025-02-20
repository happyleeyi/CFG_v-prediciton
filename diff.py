import torch
import torch.nn.functional as F
import math
import numpy as np

def logsnr_schedule_fn(t, eps=1e-12):
    """
    t: 0~1 범위의 float 또는 torch.Tensor
       alpha_t = cos(0.5*pi*t)
       sigma_t = sin(0.5*pi*t)
       snr = alpha_t^2 / sigma_t^2 = cot^2(0.5*pi*t)
       logsnr = 2 * log( cot(0.5*pi*t) )
    eps: 안전을 위한 작은 값(0 분모/로그 방지용)

    return: logsnr (t와 동일한 shape)
    """
    # cos(0.5*pi*t)와 sin(0.5*pi*t)이 0이 되지 않도록 eps로 clip

    alpha_t = torch.clip(torch.cos(0.5 * math.pi * t), min=eps, max=1-eps)
    sigma_t = torch.clip(torch.sin(0.5 * math.pi * t), min=eps, max=1-eps)

    # snr = alpha_t^2 / sigma_t^2
    # logsnr = log(alpha_t^2) - log(sigma_t^2) = 2(log(alpha_t) - log(sigma_t))
    log_alpha = torch.log(alpha_t)
    log_sigma = torch.log(sigma_t)
    logsnr = 2.0 * (log_alpha - log_sigma)

    return logsnr

def diffusion_forward(x, logsnr):
    """
    q(z_t | x).
    x, logsnr: torch.Tensor
    """
    mean = x * torch.sqrt(torch.sigmoid(logsnr))    # mean = x * sqrt(sigmoid(logsnr))
    std = torch.sqrt(torch.sigmoid(-logsnr))    # std = sqrt(sigmoid(-logsnr))
    var = torch.sigmoid(-logsnr)    # var = sigmoid(-logsnr)
    logvar = F.logsigmoid(-logsnr)  # logvar = log_sigmoid(-logsnr)

    return {
        'mean': mean,
        'std': std,
        'var': var,
        'logvar': logvar
    }

def predict_x_from_eps(z, eps, logsnr):
    """
    x = (z - sigma*eps)/alpha 
      = sqrt(1 + exp(-logsnr)) * (z - eps / sqrt(1 + exp(logsnr)))
    """
    # alpha = sqrt(sigmoid(logsnr)), sigma = sqrt(sigmoid(-logsnr))
    # 하지만 식을 직접 전개하면 다음과 같이 표현 가능
    return torch.sqrt(1. + torch.exp(-logsnr)) * (
        z - eps * (1. / torch.sqrt(1. + torch.exp(logsnr)))
    )


def predict_xlogvar_from_epslogvar(eps_logvar, logsnr):
    """
    Scale Var[eps] by exp(-logsnr).
    x_logvar = eps_logvar - logsnr
    """
    return eps_logvar - logsnr


def predict_eps_from_x(z, x, logsnr):
    """
    eps = (z - alpha*x)/sigma 
        = sqrt(1 + exp(logsnr)) * (z - x / sqrt(1 + exp(-logsnr)))
    """
    return torch.sqrt(1. + torch.exp(logsnr)) * (
        z - x * (1. / torch.sqrt(1. + torch.exp(-logsnr)))
    )


def predict_epslogvar_from_xlogvar(x_logvar, logsnr):
    """
    Scale Var[x] by exp(logsnr).
    eps_logvar = x_logvar + logsnr
    """
    return x_logvar + logsnr


def predict_x_from_v(z, v, logsnr):
    """
    alpha_t = sqrt(sigmoid(logsnr))
    sigma_t = sqrt(sigmoid(-logsnr))
    x = alpha_t * z - sigma_t * v
    """

    alpha_t = torch.sqrt(torch.sigmoid(logsnr))
    sigma_t = torch.sqrt(torch.sigmoid(-logsnr))

    return alpha_t * z - sigma_t * v


def predict_v_from_x_and_eps(x, eps, logsnr):
    """
    v = alpha_t * eps - sigma_t * x
      = sqrt(sigmoid(logsnr)) * eps - sqrt(sigmoid(-logsnr)) * x
    """
    alpha_t = torch.sqrt(torch.sigmoid(logsnr))
    sigma_t = torch.sqrt(torch.sigmoid(-logsnr))
    return alpha_t * eps - sigma_t * x

import torch
import torch.nn.functional as F

#############################################
# 아래 4개 함수는 이미 PyTorch로 변환된 버전을 사용한다고 가정
# (혹은 앞서 답변에서 정의한 버전)
#
# predict_x_from_eps
# predict_x_from_v
# predict_eps_from_x
# predict_v_from_x_and_eps
#############################################

class Diffusion:
    def __init__(self, mean_type, training_steps):
        """
        model_fn: (z, logsnr) -> model_output
        mean_type: 'eps', 'x', or 'v'
        """
        self.mean_type = mean_type
        self.training_steps = training_steps

    def run_model(self, z, logsnr, label, model_fn):
        """
        z, logsnr: torch.Tensor
        model_fn: 네트워크 추론 함수
        """
        model_output = model_fn(z, logsnr, label)

        logsnr = logsnr.view(z.shape[0], 1, 1, 1)

        # 아래 로직은, mean_type에 따라 분기해서 model_eps, model_x, model_v를 구성
        if self.mean_type == 'eps':
            model_eps = model_output
        elif self.mean_type == 'x':
            model_x = model_output
        elif self.mean_type == 'v':
            model_v = model_output
        else:
            raise ValueError(f"Unknown mean_type: {self.mean_type}")

        # get prediction of x at t=0
        if self.mean_type == 'eps':
            model_x = predict_x_from_eps(z=z, eps=model_eps, logsnr=logsnr)
        elif self.mean_type == 'v':
            model_x = predict_x_from_v(z=z, v=model_v, logsnr=logsnr)

        # get eps prediction if clipping or if mean_type != eps
        if self.mean_type != 'eps':
            model_eps = predict_eps_from_x(z=z, x=model_x, logsnr=logsnr)

        # get v prediction if clipping or if mean_type != v
        if self.mean_type != 'v':
            model_v = predict_v_from_x_and_eps(x=model_x, eps=model_eps, logsnr=logsnr)

        return {
            'model_x': model_x,
            'model_eps': model_eps,
            'model_v': model_v
        }

    def p_losses(
        self,
        model,
        x_start: torch.Tensor,     # (B, C, H, W)
        label,
        i: torch.Tensor,           # (B,)  각 샘플마다 다른 스텝 인덱스
        loss_type: str = "SNR+1"
    ) -> torch.Tensor:
        """
        Args:
            x_start:  (B, C, H, W)
            i:        (B,) 정수 텐서. 각 배치 샘플마다 다른 스텝 인덱스
            loss_type: 'eps', 'tSNR', or 'SNR+1'
        Returns:
            loss:     스칼라 로스 (Tensor 형태)
        """
        B = x_start.shape[0]
        # i가 (B,) 형태이므로, 부동소수로 변환 후 / self.num_steps
        t_val = i.float() / self.training_steps  # (B,)

        # 스케줄 함수가 (B,) 형태의 t_val을 입력받아 (B,) 형태의 logsnr 반환한다고 가정
        logsnr_t_val = logsnr_schedule_fn(t_val)  # (B,) 형태라고 가정

        # logsnr_t_val_expand shape => (B,1,1,1)
        logsnr_t_val_expand = logsnr_t_val.view(B, 1, 1, 1)

        # Forward process (q(z_t|x)) : mean, std 구하기
        # diffusion_forward가 x, logsnr 각각 (B,C,H,W)와 (B,1,1,1)를 브로드캐스팅 가능하다고 가정
        forward_t = diffusion_forward(
            x=x_start,
            logsnr=logsnr_t_val_expand
        )
        noise = torch.randn_like(x_start)   # (B, C, H, W)
        z_t = forward_t['mean'] + forward_t['std'] * noise

        model_out = self.run_model(
            z=z_t,
            logsnr=logsnr_t_val,   # 혹은 logsnr_t_val_expand
            label=label,
            model_fn=model
        )

        # 이제 loss 계산
        if loss_type == 'eps':
            # model_out['model_eps']도 (B,C,H,W) 형태라고 가정
            # 기본 smooth_l1_loss는 reduction='mean'이라 최종 스칼라
            loss = F.smooth_l1_loss(noise, model_out['model_eps'])

        elif loss_type == 'tSNR':
            # 두 가지 로스( \epsilon -로스, x-로스 ) 각각 스칼라로 계산 후 max
            loss_eps = F.smooth_l1_loss(noise, model_out['model_eps'])
            loss_x   = F.smooth_l1_loss(x_start, model_out['model_x'])
            loss = torch.max(loss_eps, loss_x)

        elif loss_type == 'SNR+1':
            # v 타겟: predict_v_from_x_and_eps(x, eps, logsnr)
            v_target = predict_v_from_x_and_eps(
                x=x_start,
                eps=noise,
                logsnr=logsnr_t_val_expand
            )
            loss = F.smooth_l1_loss(v_target, model_out['model_v'])

        else:
            raise NotImplementedError(f"Unknown loss_type: {loss_type}")

        return loss
    
    def ddim_step(self, model, i, z_t, num_steps, label, w):
        """
        i: 현재 스텝 (int)
        z_t: 현재 스텝에서의 노이즈가 추가된 상태 (torch.Tensor)
        num_steps: 전체 DDIM 스텝 수
        logsnr_schedule_fn: (time_float) -> logsnr 값을 반환하는 스케줄 함수
        """
        shape = z_t.shape
        # z_t.dtype와 z_t.device를 사용하여 동일한 dtype, device를 가진 텐서를 생성
        dtype = z_t.dtype
        device = z_t.device

        # i+1, i를 각각 / num_steps 하여 time 구하기
        t_val = (i + 1.) / num_steps
        s_val = i / num_steps

        # 스케줄 함수로부터 스칼라(또는 텐서) logsnr 값 구하기
        logsnr_t_val = logsnr_schedule_fn(t_val)   # 예: float 또는 torch.Tensor
        logsnr_s_val = logsnr_schedule_fn(s_val)

        # batch 차원(shape[0])만큼 동일한 값을 채운 텐서 생성
        # (원본 코드: jnp.full((shape[0],), logsnr_t))
        logsnr_t = torch.full((shape[0],), fill_value=logsnr_t_val,
                              dtype=dtype, device=device)
        logsnr_s = torch.full((shape[0],), fill_value=logsnr_s_val,
                              dtype=dtype, device=device)
        label = torch.full((shape[0],), fill_value=label,
                              dtype=int, device=device)


        # 모델로부터 x, eps, v 예측
        model_out_none = self.run_model(
            z=z_t,
            logsnr=logsnr_t,
            label=None,
            model_fn=model
        )
        model_out = self.run_model(
            z=z_t,
            logsnr=logsnr_t,
            label=label,
            model_fn=model
        )
        #eps_pred_t = (1+w)*model_out['model_eps']-w*model_out_none['model_eps']                CFG 논문 형태
        eps_pred_t = model_out_none['model_eps'] + w * (model_out['model_eps'] - model_out_none['model_eps'])       #stable diffusion에 사용되는 CFG 형태 - 더 안정적임
        x_pred_t = predict_x_from_eps(z_t, eps_pred_t, logsnr_t)

        # stdv_s = sqrt(sigmoid(-logsnr_s))
        stdv_s = torch.sqrt(torch.sigmoid(-logsnr_s))
        # alpha_s = sqrt(sigmoid(logsnr_s))
        alpha_s = torch.sqrt(torch.sigmoid(logsnr_s))

        # z_s_pred = alpha_s * x_pred_t + stdv_s * eps_pred_t
        z_s_pred = alpha_s * x_pred_t + stdv_s * eps_pred_t

        # i == 0이면 x_pred_t(더 이상 역노이징 불필요), 그렇지 않으면 z_s_pred
        if i == 0:
            return x_pred_t
        else:
            return z_s_pred

    def sample_images(self, model, n_images, num_steps, channels, image_size, label, w, store_each_step=False, device='cuda'):
        """
        Args:
            model: DDIM Model 객체. (model.ddim_step(i, z_t, num_steps, fn) 사용)
            n_images: 생성할 이미지 개수 (batch 크기)
            num_steps: DDIM total steps
            logsnr_schedule_fn: t (0~1 float) -> logsnr 값을 반환하는 함수
            shape: z_t 초기 텐서의 shape (C, H, W) 같은 형태
            store_each_step: True면 모든 스텝의 이미지를 저장, False면 최종 결과만 저장
            device: GPU or CPU
        Returns:
            samples_list: 생성된 이미지를 담은 리스트
                        store_each_step=True이면
                            각 (n_images)마다 [z_t(step=num_steps-1), ..., z_t(step=0)] 모두 저장
                        store_each_step=False이면
                            각 (n_images)마다 [z_t(step=0)]만 저장
        """

        # 결과를 저장할 전체 리스트
        # 내부 구조 예시 (store_each_step=True 가정):
        # [
        #   [img_stepN, img_stepN-1, ..., img_step0],  # 1번째 이미지 생성 과정
        #   [img_stepN, img_stepN-1, ..., img_step0],  # 2번째 이미지 생성 과정
        #   ...
        # ]
        samples_list = []

        # DDIM 샘플링은 일반적으로 가우시안 노이즈로부터 시작
        # 예: torch.randn으로 초기화
        for img_idx in range(n_images):
            # shape: (C,H,W)
            # => 배치 차원을 추가: (1, C, H, W)
            z_t = torch.randn((1, channels, image_size, image_size), device=device)

            # 이번 이미지에서 스텝별 결과를 담을 리스트
            # (store_each_step=True가 아닐 경우, 마지막 샘플만 보관)
            step_results = []

            # t=1부터 t=0까지 => 스텝 인덱스가 1에서 0이 되려면,
            # num_steps개 중 마지막 인덱스를 0으로 보고, 거꾸로 내려감
            # 예) num_steps=50이면 i는 49 -> 0
            for i in reversed(range(num_steps)):
                # ddim_step 호출
                # z_t 갱신
                print('sampling '+str(i)+'step')
                i = torch.as_tensor(i)
                z_t = self.ddim_step(
                    model=model,
                    i=i,
                    z_t=z_t,
                    num_steps=num_steps,
                    label=label,
                    w=w
                )
                # 스텝마다 결과 저장 또는 건너뛰기
                if store_each_step:
                    # (1, C, H, W) 텐서를 복사하여 저장 (clone / detach 등)
                    step_results.append(z_t.clone().detach().cpu().numpy())

            # 모든 스텝을 돌고 나면, z_t는 t=0 이미지가 됨
            if not store_each_step:
                # 최종 결과만 저장
                step_results.append(z_t.clone().detach().cpu().numpy())

            # step_results: (스텝 개수 개 or 1개)의 (1, C, H, W)
            # 원하는 대로 list화
            samples_list.append(step_results)

        # 최종: 각 이미지마다의 스텝별 결과가 담긴 리스트들의 리스트
        return np.array(samples_list)
