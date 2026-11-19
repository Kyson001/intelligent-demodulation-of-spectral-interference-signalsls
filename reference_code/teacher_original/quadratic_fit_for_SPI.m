%色散干涉法 的仿真 无色散，等频率间隔。非常重要

clc
clear all;
% close all;
format long  %将数值显示格式设置为长固定十进制小数点格式

%%
%参数设定
c=3E8;                         %光速
deltaf = 50e9;                 %频率间隔50GHZ
f1 = 1.85e14;                  %频率，单位Hz
lambda1 = c*1e9/f1;            %波长，单位nm
f2 = 2.00e14; 
lambda2 = c*1e9/f2;
f = f1 + deltaf : deltaf : f2;  %采样点频率
waveband = lambda1 - lambda2; B0=f2-f1;   %频率带宽
w = 2 * pi .*f;    %每个频率对应角频率
lambda = c ./ f;   %每个频率对应波长
fs = 1 / deltaf;   %等价采样率，计算为20pHz
N = B0 / deltaf;    %采样点数，N= 2300
df= fs / N;        %FFt的频率分辨力
Length = length(f);                  %FFT的长度
n = Length;
ft=[0:df:df*(n-1)] - fs/2;%时间轴
%%
%生成高斯光谱外形的光频梳
% I0=gaussmf(f,[0.021e14 1.91e14])+0.8*gaussmf(f,[0.02e14 1.99e14]);
Icomb = gaussmf(f,[0.04e14 1.925e14]);   %光频梳包络为高斯函数
figure                                   %光频梳形状
subplot(2,1,1);
plot(f,Icomb);
xlabel('Frequency');
ylabel('Power');
title('OFC_Spectrum');
%%
reso_t = 1/B0;                %时间分辨力
reso_L = c * reso_t / 2;       %分辨力


%% 干涉模型
a = 0.5; b = 0.5;             %光强比
for j = 1 : 1 : 20
 L=j* 0.5 *1e-3;               %若干待测距离mm
%L = 0.5e-3; 
%L = 1e-3; 
%L = 1.5e-3; 
%L = 2e-3; 
%L = 2.9e-3;
for i = 1:1:N                 %求解光程差对应的相移
phase(i) = w(i)* L/c;
end
Idpi = Icomb.*(a^2+b^2+2*a*b.*cos(phase));    %干涉谱
subplot(2,1,2);
plot(f,Idpi);
xlabel('Frequency');
ylabel('Power');
title('DPI_Spectrum');

%%
% 计算 FFT
fft_result = fft(Idpi);       % FFT 变换
magnitude_spectrum = abs(fft_result/n); % 计算幅度谱及归一化
magnitude_spectrum = magnitude_spectrum(1:n/2+1); % 取单边频谱
magnitude_spectrum(2:end-1) = 2 * magnitude_spectrum(2:end-1); % 修正幅度=原幅度*2

% 时间向量
ft = (0:(n/2)) * (fs/n);   %0~fs/2
% 寻峰
[peaks, locs] = findpeaks(magnitude_spectrum, 'MinPeakHeight', 0.02); % 寻峰并设置最小峰值高度

window_size = 5; % 设置峰值附近的窗口大小
fit_fts = zeros(1, length(locs)); % 存储二项式拟合计算的时间


for i = 1:length(locs)
    % 确定峰值附近的区域
    start_idx = max(1, locs(i) - window_size);   %141
    end_idx = min(length(ft), locs(i) + window_size);  %151
    
    % 提取区域内的时间和幅度
    ft_window = ft(start_idx:end_idx);
    magnitude_window = magnitude_spectrum(start_idx:end_idx);
    
    % 二项式拟合
    p = polyfit(ft_window, magnitude_window, 2);  %二项式拟合
    y = p(1) .* ft_window .^2 + p(2) .* ft_window + p(3);
    % 输出结果
    fit_fts(i) = -p(2) / (2 * p(1));    %峰值坐标：x=-b/(2a)
end

% 输出结果
result_quadratic = zeros(2,8);
disp('Peaks found at (Quadratic_fit method):');
 fprintf('Peak: %.9f s\n',  fit_fts);
L1 = c * fit_fts * 1e3; Accu = L1 - 0.3 * j; 
fprintf('Lmea=%.9f mm Accu=%.9f mm\n', L1, Accu);
% result_quadratic (1,j) = L1;
% result_quadratic (2,j) = Accu;
hold on
end
%二项式拟合法
x1 = [6.600000000000000e-12, 6.666666666666667e-12, 6.733333333333333e-12];
y1 = [0.096401949993566, 0.313904747943254, 0.096402004286866];
p = polyfit(x1,y1,2);
pl = -p(2)/(2*p(1));
x2 = linspace(6.600000000000000e-12,6.733333333333333e-12,100);
y2 = p(1) .* x2 .^2 + p(2) .*x2 + p(3);
% 绘制时域脉冲
figure;
set(gcf,'color','White')
g = plot(ft, magnitude_spectrum);
set(g(1),'LineWidth',2)
hold on;
plot(fit_fts, peaks, 'ro', 'MarkerFaceColor', 'r'); % 标记峰值
xlabel('Time (s)');
set(gca,'FontSize',16)
ylabel('Magnitude');
set(gca,'FontSize',16)
title('Time-domain Peaks Location with Quadratic Polyfit Method');
set(gca,'FontSize',20)
%plot(ft_window,y,'r-');
hold on
h = plot(x2,y2,'r--');
set(h(1),'LineWidth',2)
legend('L=1 mm')
grid on;
%end

