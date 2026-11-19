import numpy as np

lambda_nm = np.arange(1510, 1591, 1)
lambda_m = lambda_nm * 1e-9

def generate_spectrum(d):
%     #d:待测位移,单位 m
    delta_L = 2 * d
    lam_c = 1550e-9
   %随机包络系数
    a1 = np.random.uniform(-8e-4, 8e-4)
    a2 = np.random.uniform(-2e-6, 2e-6)
    A = 1 + a1*(lambda_m - lam_c) + a2*((lambda_m - lam_c) **2)
    A = A / np.max(np.abs(A))
    %对比度
    gamma = np.random.uniform(0.4, 0.95)
    %相位
    phi = 2 * np.pi * delta_L / lambda_m;
    phi += np.random.uniform(-0.08*np.pi, 0.08*np.pi, size=phi.shape)
    %干涉强度
    S = A * (1 + gamma * np.cos(phi))
    %噪声
    snr_db = np.random.uniform(5, 40)
    signal_power = np.mean(S**2)
    noise_power = signal_power / (10**(snr_db/10))
    noise = np.random.normal(0, np.sqrt(noise_power), size=S.shape)
    S = S + noise
    return S