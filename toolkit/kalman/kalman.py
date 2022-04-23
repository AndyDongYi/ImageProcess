# -- coding: utf-8 --
'''
--------------------------------------------------------------------
为了帮助大家理解滤波器的相关实现方法，构建了一份一维滤波器样例代码
kalman，extend kalman，uscent kalman和 particle filter
多维滤波器与一维原理一致，代码供大家参考
有问题可联系 yi.dong@horizon.ai
--------------------------------------------------------------------
'''

import numpy as np
import matplotlib.pyplot as plt
import math
import os
N = 200
Q = 0.1
R = 1
A = 0.5
H = 3
B = 1

#绘制图形参数
SubPlotCount = 0
TotalSubPlotNum = 4

#生成高斯噪声，分别为均值为0，方差为Q和R的高斯分布
predict_gauss_noise = np.random.normal(0,Q**0.5,N)
measure_gauss_noise = np.random.normal(0,R**0.5,N)
#plt.plot(measure_gauss_noise,label='measure_gauss_noise')
#plt.plot(predict_gauss_noise,label='predict_gauss_noise')

#产生测量数据和真值模拟数据
def GenerateTestData(N,measure_gauss_noise,kalman_predict_func,kalman_measure_func):
    #为了区分滤波器的初始值，真值初始化状态设置为3，滤波器的初始化状态设置的是0
    Xk = 3
    TrueData = []
    MeasureData = []
    for i in range(N):
        Xk = kalman_predict_func(Xk)
        TrueData.append(Xk)
        # 观测值Zk = H*Xpre + R
        Zk = kalman_measure_func(Xk,measure_gauss_noise[i])
        MeasureData.append(Zk)
    return TrueData,MeasureData

def DrawKalmanPlot(kalman_estimate,label_kalman_estimate,measure_transform_data,label_measure_transform_data,true_data
                   ,label_true_data,predict_estimate,lable_predict_estimate):
    global SubPlotCount
    SubPlotCount = SubPlotCount + 1
    ax = plt.subplot(TotalSubPlotNum,1,SubPlotCount)
    plt.sca(ax)
    plt.plot(kalman_estimate,label=label_kalman_estimate)
    #plt.plot(predict_estimate,label=lable_predict_estimate)
    plt.plot(true_data,label=label_true_data)
    plt.plot(measure_transform_data,label=label_measure_transform_data)
    plt.legend(loc=1,fontsize=6)

#卡尔曼预测和测量函数
kalman_predict_func = lambda x:A*x + B
kalman_measure_func = lambda x,measure_guass_noise:H*x + measure_guass_noise

def KalmanFilter(predict_gauss_noise,Xpost,Ppost,measure_data,Q,R,A,H,kalman_predict_func):
    #状态转移方程更新 x(k) = 0.5*x(k-1) + 1*u(k-1) + Q，假设u(k-1)为1
    Xpre =  kalman_predict_func(Xpost) + predict_gauss_noise
    #先验估计方差更新Ppre = APpostA' + Q,传递的高斯噪声为标准正态分布
    Ppre = A*Ppost*A + Q
    #卡尔曼增益Kk = Ppre*H'/(H*Ppre*H'+R),R为测量噪声
    Kk = Ppre*H/(H*Ppre*H + R)
    #观测方程更新
    Xpost = Xpre + Kk*(measure_data - H*Xpre)
    Ppost = (1 - Kk*H)*Ppre
    return Xpost,Ppost,Xpre

def RunKalmanFilter():
    #Kalman Filter Data
    #卡尔曼滤波器的最优状态估计和方差，初始化设置的0
    Xpost = 0
    Ppost = 0.0
    kalman_estimate = []
    kalman_predict_estimate = []
    kalman_test_data = GenerateTestData(N, measure_gauss_noise,kalman_predict_func,kalman_measure_func)
    kalman_true_data = kalman_test_data[0]
    kalman_measure_data = kalman_test_data[1]
    kalman_measure_transform_data = list( data/H for data in kalman_measure_data)
    for i in range(N):
        kalman_data =  KalmanFilter(predict_gauss_noise[i],Xpost,Ppost,kalman_measure_data[i],Q,R,A,H,kalman_predict_func)
        Xpost = kalman_data[0]
        Ppost = kalman_data[1]
        Xpre = kalman_data[2]
        kalman_estimate.append(Xpost)
        kalman_predict_estimate.append(Xpre)
    DrawKalmanPlot(kalman_estimate,'kalman_filter_estimate',kalman_measure_transform_data,'measure_transform_data',
                   kalman_true_data,'test_true_data',kalman_predict_estimate,'kalman_predict_estimate')

#扩展卡尔曼预测和测量函数
extend_predict_func = lambda x:A*x*(math.sin(x)) + B*x
extend_measure_func = lambda x,measure_guass_noise:0.05*H if x <= 0.1 else H*x**0.5 + measure_guass_noise
jacobian_a_func = lambda x:A * x * (math.cos(x)) + B
jacobian_h_func = lambda x: 0 if x <= 0.1 else 0.5 * H / (x ** 0.5)

def ExtendKalmanFilter(predict_gauss_noise,Xpost,Ppost,measure_data,Q,R,jacobian_a_func,jacobian_h_func,kalman_predict_func):
    #状态转移方程更新 x(k) = 0.5*x(k-1) + 1*u(k-1) + Q，假设u(k-1)为1
    Xpre =  kalman_predict_func(Xpost) + predict_gauss_noise
    #先验估计方差更新Ppre = APpostA' + Q,传递的高斯噪声为标准正态分布，Q为1
    Ppre = jacobian_a_func(Xpost)*Ppost*jacobian_a_func(Xpost) + Q
    #卡尔曼增益Kk = Ppre*H'/(H*Ppre*H'+R),H为jacobian_h_func,R为测量噪声
    Kk = Ppre*jacobian_h_func(Xpre)/(jacobian_h_func(Xpre)*Ppre*jacobian_h_func(Xpre) + R)
    #观测方程更新
    Xpost = Xpre + Kk*(measure_data - jacobian_h_func(Xpre)*Xpre)
    Ppost = (1 - Kk*jacobian_h_func(Xpre))*Ppre
    return Xpost,Ppost,Xpre

def RunExtendKalmanFilter():
    #Extend Kalman Filter Data
    # 扩展卡尔曼滤波器的最优状态估计和方差，初始化设置的0
    extend_Xpost = 0.0
    extend_Ppost = 0.0
    extend_kalman_estimate = []
    extend_kalman_predict_estimate = []
    extend_kalman_test_data = GenerateTestData(N, measure_gauss_noise,extend_predict_func,extend_measure_func)
    extend_kalman_true_data = extend_kalman_test_data[0]
    extend_kalman_measure_data = extend_kalman_test_data[1]
    extend_kalman_measure_transform_data = list((data/H)**2 for data in extend_kalman_measure_data)
    for i in range(N):
        extend_kalman_data = ExtendKalmanFilter(predict_gauss_noise[i], extend_Xpost, extend_Ppost, extend_kalman_measure_data[i], Q, R, jacobian_a_func,jacobian_h_func,extend_predict_func)
        extend_Xpost = extend_kalman_data[0]
        extend_Ppost = extend_kalman_data[1]
        extend_Xpre = extend_kalman_data[2]
        extend_kalman_estimate.append(extend_Xpost)
        extend_kalman_predict_estimate.append(extend_Xpre)
    DrawKalmanPlot(extend_kalman_estimate,"extend_kalman_estimate",extend_kalman_measure_transform_data,"measure_transform_data"
                   ,extend_kalman_true_data,"test_true_data",extend_kalman_predict_estimate,"extend_kalman_predict_estimate")

#无迹卡尔曼预测和测量函数
uscent_predict_func = lambda x:A*x*(math.sin(x)) + B*x
uscent_measure_func = lambda x,measure_guass_noise:0 if x < 0.01 else H*x**0.5 + measure_guass_noise

def UscentKalmanFilter(Xpost,Ppost,ramda,uscent_predict_func,uscent_measure_func,Q,R,measure_data,WeightMean,WeightConvance,measure_gauss_noise,predict_gauss_noise):
    SigmaPredict = []
    Xpredict = []
    Xexpect = 0
    ConX = 0
    Yupdate = []
    ConY = 0
    ConXY = 0
    Yexpect = 0
    SigmaPost = []
    #sigma点采样，用于Predict过程
    SigmaPredict.append(Xpost)
    SigmaPredict.append(Xpost + math.sqrt((1+ ramda)*Ppost))
    SigmaPredict.append(Xpost - math.sqrt((1 + ramda) * Ppost))
    #sigma点预测值,期望及方差计算
    for value in SigmaPredict:
        Xpredict.append(uscent_predict_func(value) + predict_gauss_noise)
    for i in range(len(Xpredict)):
        Xexpect = Xexpect + Xpredict[i] * WeightMean[i]
    for i in range(len(Xpredict)):
        ConX = ConX + WeightConvance[i]*(Xpredict[i] - Xexpect)**2
    ConX = ConX + Q
    #sigma点采样，用于Update过程,多维的话需要用到cholesky分解
    SigmaPost.append(Xexpect)
    SigmaPost.append(Xexpect + math.sqrt((1 + ramda) * ConX))
    SigmaPost.append(Xexpect - math.sqrt((1 + ramda) * ConX))
    #sigma点观测值,期望及方差计算
    for value in SigmaPost:
        Yupdate.append(uscent_measure_func(value,measure_gauss_noise))
    for i in range(len(Yupdate)):
        Yexpect = Yexpect + Yupdate[i] * WeightMean[i]
    #计算Y及XY协方差
    for i in range(len(Yupdate)):
        ConY = ConY + WeightConvance[i]*(Yupdate[i] - Yexpect)**2
        ConXY = ConXY + WeightConvance[i]*(Xpredict[i] - Xexpect)*(Yupdate[i] - Yexpect)
    #计算卡尔曼增益,最优估计及方差更新
    Kk = ConXY/(ConY + R)
    Xpost = Xexpect + Kk*(measure_data - Yexpect)
    Ppost = ConX - Kk*ConY*Kk
    return Xpost,Ppost,Xexpect

def RunUscentKalmanFilter(ramda=1,alpha=0,k=0,belta=0):
    WeightMean = [ramda/(1 + ramda), 1/(2*(1 + ramda)), 1/(2*(1 + ramda))]
    WeightConvance = [ramda/(1 + ramda), 1/(2*(1 + ramda)), 1/(2*(1 + ramda))]
    # 卡尔曼滤波器的最优状态估计和方差，初始化设置的0
    uscent_Xpost = 0
    uscent_Ppost = 0
    uscent_kalman_estimate = []
    uscent_kalman_predict_estimate = []
    uscent_kalman_test_data = GenerateTestData(N, measure_gauss_noise, uscent_predict_func, uscent_measure_func)
    uscent_kalman_true_data = uscent_kalman_test_data[0]
    uscent_kalman_measure_data = uscent_kalman_test_data[1]
    uscent_kalman_measure_transform_data = list((data / H) ** 2 for data in uscent_kalman_measure_data)
    for i in range(N):
        uscent_kalman_data = UscentKalmanFilter(uscent_Xpost,uscent_Ppost,ramda,uscent_predict_func,uscent_measure_func,Q,R,uscent_kalman_measure_data[i],WeightMean,WeightConvance,measure_gauss_noise[i],predict_gauss_noise[i])
        uscent_Xpost = uscent_kalman_data[0]
        uscent_Ppost = uscent_kalman_data[1]
        uscent_Xexpect = uscent_kalman_data[2]
        uscent_kalman_estimate.append(uscent_Xpost)
        uscent_kalman_predict_estimate.append(uscent_Xexpect)
    DrawKalmanPlot(uscent_kalman_estimate, "uscent_kalman_estimate",uscent_kalman_measure_transform_data, "measure_transform_data",
                   uscent_kalman_true_data,"test_true_data",uscent_kalman_predict_estimate,"uscent_kalman_predict_estimate")

#粒子滤波
particle_predict_func = lambda x: A*x*(math.sin(x)) + B*x
particle_measure_func = lambda x,measure_guass_noise:0 if x < 0.01 else H*x**0.5 + measure_guass_noise
probability_density_func = lambda x,u,sigma: 1/(math.sqrt(2*(math.pi))*sigma)*math.exp((-1*(x - u)**2)/(2*(sigma**2)))

def ParticleFilter(Xparticle,measure_data,particle_predict_func,particle_measure_func,probability_density_func):
    Yupdate = []
    Weight = []
    Xpre_particle = []
    Xpost_particle = []
    # 更新粒子状态,np.random.normal每个筛选后粒子的差异
    for value in Xparticle:
        predict_gauss_noise = np.random.normal(0, Q ** 0.5)
        Xpre_particle.append(particle_predict_func(value) + predict_gauss_noise)
    for value in Xpre_particle:
        measure_gauss_noise = np.random.normal(0, Q ** 0.5)
        Yupdate.append(particle_measure_func(value, measure_gauss_noise))
    for value in Yupdate:
        Weight.append(probability_density_func(measure_data,value,1))
    # 计算粒子归一化权重
    for i in range(len(Weight)):
        Weight[i] = Weight[i]/np.sum(Weight)
    #重采样
    for i in range(len(Weight)):
        rand = np.random.uniform(0,max(Weight))
        for j in range(len(Weight)):
            if (rand <= Weight[j]):
                Xpost_particle.append(Xpre_particle[j])
                break
    #计算估计均值
    Xpost = np.sum(Xpost_particle)/len(Xpost_particle)
    #print("Xpost:",Xpost,"len(Weight):",len(Weight),"len(Xpost_particle):",len(Xpost_particle))
    return Xpost,Xpost_particle

def RunParticleFilter(particle_num = 100):
    particle_Ppost = 0
    particle_kalman_estimate = []
    particle_kalman_predict_estimate = []
    particle_kalman_test_data = GenerateTestData(N, measure_gauss_noise, particle_predict_func, particle_measure_func)
    particle_kalman_true_data = particle_kalman_test_data[0]
    particle_kalman_measure_data = particle_kalman_test_data[1]
    particle_kalman_measure_transform_data = list((data / H) ** 2 for data in particle_kalman_measure_data)
    #产生粒子，初始化粒子分布为【10，100】的均匀分布，初始值的选取影响的是收敛速度，极端情况下会造成滤波发散
    Xparticle = np.random.uniform(10,100,particle_num)
    for i in range(N):
        particle_kalman_data = ParticleFilter(Xparticle,particle_kalman_measure_data[i],particle_predict_func,
                                            particle_measure_func,probability_density_func)
        particle_Xpost = particle_kalman_data[0]
        Xparticle = particle_kalman_data[1]
        particle_Xexpect = particle_predict_func(particle_Xpost) + predict_gauss_noise[i]
        particle_kalman_estimate.append(particle_Xpost)
        particle_kalman_predict_estimate.append(particle_Xexpect)
    DrawKalmanPlot(particle_kalman_estimate, "particle_filter_estimate", particle_kalman_measure_transform_data,
                   "measure_transform_data",
                   particle_kalman_true_data, "test_true_data", particle_kalman_predict_estimate,
                   "particle_filter_predict_estimate")


if __name__ == '__main__':
    RunKalmanFilter()
    RunExtendKalmanFilter()
    RunUscentKalmanFilter(1)
    RunParticleFilter()
    #plt.savefig(os.getcwd() + r"\filter.png")
    plt.show()