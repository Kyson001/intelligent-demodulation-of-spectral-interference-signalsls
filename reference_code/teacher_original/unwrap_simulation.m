%色散干涉法 的仿真 无色散，等频率间隔。非常重要

clc
%clear all;
% close all;
%format long  %将数值显示格式设置为长固定十进制小数点格式

% datawavelength = linspace(1500,1600,512);%设定波长范围nm
% tex=csvread('ceshi69.csv');%读取光谱数据
%%
%参数设定
c=3E8;%光速
% deltal=0.5;%波长间隔
    f1500=c*1e9/1500;%1500对应200THz
    f1550=c*1e9/1550;
    f1600=c*1e9/1600;
% deltaf=10e9*(c*deltal)/(1550^2);%等频率间隔转化
deltaf=250e6;%频率GHZ
deltal=(c*deltaf)/(f1550^2);%1550附近的波长间隔
fbegain=1.85e14; wavebegain=c*1e9/fbegain;
fend=2.00e14; waveend=c*1e9/fend;
f=fbegain+deltaf:deltaf:fend;%采样点频率
waveband=wavebegain-waveend; B0=fend-fbegain;%带宽
w=2*pi*f;
wave=c./f;
fs=1/deltaf;%等价采样率
N=B0/deltaf;%点数
df=fs/N;%FFt的频率分辨力
lambda = 1550e-9;
%%
%生成高斯光谱外形的光频梳
% I0=gaussmf(f,[0.021e14 1.91e14])+0.8*gaussmf(f,[0.02e14 1.99e14]);
I0=gaussmf(f,[0.04e14 1.925e14]);
%figure
%subplot(3,2,1);
%plot(f,I0);
%xlabel('Frequency');
%ylabel('Power');
%title('OFC_Spectrum');
% figure
% % subplot(1,3,1);
% plot(f,I0);
a1=0.5; b1=0.5;%光强比
deltat=1/B0;%sj分辨力
deltaL=c/B0/2;%分辨力
Length=length(f);N=Length;
ft=[0:df:df*(N-1)] - fs/2;%时间轴
L_save = zeros(1,100);
%for j=1:1:5
%L=(j*5000+100.5)*1e-3;%带测距离mm；
for j=1:1:10
    Ltarget=j* 0.3 *1e-3;%带测距离mm；
%L1 = 0.5e-3;
%Ltarget = 2.9e-3;
    for i=1:1:N
%phi(i)=ng(wave(i)*1e9)*w(i)*L/c;   %不太理解为啥乘波长
        phi(i) = w(i)* Ltarget/c;
    end
    I1=I0 .*(a1^2+b1^2+2*a1*b1.*cos(phi));
%subplot(3,2,2);
%plot(f,I1);
%xlabel('Frequency');
%ylabel('Power');
%title('DPI_Spectrum');

%%
%FFT
    dataFFT=fft(I1,Length)/Length;       %归一化，零频在fft开始变换的位置
    dataFFT=fftshift(dataFFT); 
    subplot(2,1,1);
    plot(abs(real(dataFFT)));  
    xlabel('Time')
    ylabel('Power')
%1.5mm显示峰值在1.51mm，误差为10微米
%将零频率分量移到频谱中心的位置，形成关于零频对称的频谱
    dataFFT(1:(N-1)/2+5)=0; %滤波包络

    iFFT=ifft(ifftshift(dataFFT),N);     %将相位信号转换到频域
    phim=angle(iFFT);                    %angle函数在MATLAB中用于计算复数的相位角. 当有一个复数数组时，angle函数可以为每个元素返回其相位角，这个角度的范围在[-π,π]之间。
    phiu=unwrap(phim);                   %相位解卷裹，起到平滑曲线的作用，让相位值显示为真值
    % plot(phi-phi(1));
    % hold on
    subplot(2,1,2);
    %scatter(f,phiu);    %1mm测量斜率为k=1.0473,则所测距离为L=kc/(2*pi)=0.5003025,精度为0.3微米
    %hold on;
    plot(f,phiu);
    xlabel('Frequency (Hz)')
    ylabel('Power')
    p = polyfit(f,phiu,2);
    p1 = p(1);
    p2 = p(2);
    %2mm测量值2.0946mm，误差为94.6微米
    %1.5mm测量值为1.5699mm，误差为69.9微米
    d_mea = p2 * c / (2 * pi) * 1e3;
    L_save = d_mea;
    Accuracy = d_mea - 0.3.*j;
    % 输出结果
    fprintf('Measured distance=%.9f mm, Accuracy = %.9f mm\n', d_mea, Accuracy);
end
% L_set = linspace(0,3,100);
% plot(L_set,L_save);