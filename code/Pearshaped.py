# elasitc
import math
import torch
import torch.nn as nn
import numpy as np
from pyDOE import lhs
from torch.utils.tensorboard import SummaryWriter
import os
import pandas as pd
import matplotlib.pyplot as plt
from idaes.core.surrogate.pysmo.sampling import HammersleySampling
import shutil

torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
np.random.seed(1234)
torch.set_default_tensor_type(torch.DoubleTensor)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"use {device} to compute")
writer = SummaryWriter('./path/to/log/Pear')

E = 10  # Young's modulus
miu = 0.3  # Poisson's ratio


class DNN1(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, hidden_num):
        super(DNN1, self).__init__()
        self.net = nn.Sequential()
        self.net.add_module('Linear_layer_1', nn.Linear(input_size, hidden_size))
        self.net.add_module('Tanh_layer_1', nn.Tanh())
        for num in range(2, hidden_num + 1):
            self.net.add_module('Linear_layer_%d' % (num), nn.Linear(hidden_size, hidden_size))  # Linear layer
            self.net.add_module('Tanh_layer_%d' % (num), nn.Tanh())  # Activation Layer
        self.net.add_module('Linear_layer_final', nn.Linear(hidden_size, output_size))  # Output Layer

    # Forward Feed
    def forward(self, x):
        return self.net(x)


class DNN2(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, hidden_num):
        super(DNN2, self).__init__()
        self.net = nn.Sequential()
        self.net.add_module('Linear_layer_1', nn.Linear(input_size, hidden_size))
        self.net.add_module('Tanh_layer_1', nn.Tanh())
        for num in range(2, hidden_num + 1):
            self.net.add_module('Linear_layer_%d' % (num), nn.Linear(hidden_size, hidden_size))  # Linear layer
            self.net.add_module('Tanh_layer_%d' % (num), nn.Tanh())  # Activation Layer
        self.net.add_module('Linear_layer_final', nn.Linear(hidden_size, output_size))  # Output Layer

    # Forward Feed
    def forward(self, x):
        return self.net(x)


class DNN3(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, hidden_num):
        super(DNN3, self).__init__()
        self.net = nn.Sequential()
        self.net.add_module('Linear_layer_1', nn.Linear(input_size, hidden_size))
        self.net.add_module('Tanh_layer_1', nn.Tanh())
        for num in range(2, hidden_num + 1):
            self.net.add_module('Linear_layer_%d' % (num), nn.Linear(hidden_size, hidden_size))  # Linear layer
            self.net.add_module('Tanh_layer_%d' % (num), nn.Tanh())  # Activation Layer
        self.net.add_module('Linear_layer_final', nn.Linear(hidden_size, output_size))  # Output Layer

    # Forward Feed
    def forward(self, x):
        return self.net(x)


class ConcatNet(nn.Module):
    def __init__(self, net1, net2, net3):
        super(ConcatNet, self).__init__()
        self.net1 = net1
        self.net2 = net2
        self.net3 = net3

    def forward(self, x):
        out1 = self.net1(x)
        out2 = self.net2(x)
        out3 = self.net3(x)
        out = torch.cat((out1, out2), dim=1)
        out = torch.cat((out, out3), dim=1)
        # out = torch.cat((out, out4), dim=1)
        return out


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    # 也可以判断是否为conv2d，使用相应的初始化方式
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    # 是否为批归一化层
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class PINN:
    def __init__(self):
        net1 = DNN1(input_size=2, hidden_size=60, output_size=2, hidden_num=6).to(device)
        net2 = DNN2(input_size=2, hidden_size=90, output_size=3, hidden_num=6).to(device)
        net3 = DNN3(input_size=2, hidden_size=40, output_size=1, hidden_num=6).to(device)
        self.model = ConcatNet(net1, net2, net3)
        ### Use DataParallel ###
        # if torch.cuda.device_count() > 1:
        # print("Let's use", torch.cuda.device_count(), "GPUs!")
        # self.model = nn.DataParallel(self.model)
        self.model.apply(weight_init)
        print(self.model)
        # data gernerate
        bounds = [[-0.5, -0.5], [0.5, 0.5]]
        init_data = HammersleySampling(data_input=bounds, number_of_samples=40000)
        data_c = init_data.sample_points()
        x_c = data_c[:, 0]
        y_c = data_c[:, 1]
        X_c = np.hstack((x_c.reshape(-1, 1), y_c.reshape(-1, 1)))

        x_ub = -0.5 + 1 * lhs(1, 200)
        y_ub = 0.5 * np.ones((200,))
        x_lb = -0.5 + 1 * lhs(1, 200)
        y_lb = -0.5 * np.ones((200,))
        X_ub = np.hstack((x_ub.reshape(-1, 1), y_ub.reshape(-1, 1)))
        X_lb = np.hstack((x_lb.reshape(-1, 1), y_lb.reshape(-1, 1)))

        t = 1 * np.ones_like(x_ub)
        t = t.reshape(-1, 1)

        x_leftb = -0.5 * np.ones((200,))
        y_leftb = -0.5 + 1 * lhs(1, 200)
        X_leftb = np.hstack((x_leftb.reshape(-1, 1), y_leftb.reshape(-1, 1)))
        x_rightb = 0.5 * np.ones((200,))
        y_rightb = -0.5 + 1 * lhs(1, 200)
        X_rightb = np.hstack((x_rightb.reshape(-1, 1), y_rightb.reshape(-1, 1)))
        zerobd = np.zeros((200, 1))

        X_lb2 = torch.tensor((-0.5, -0.5)).reshape(1, 2)
        self.X_lb2 = X_lb2
        self.X_lb2 = self.X_lb2.to(device)
        lb2 = torch.zeros((1,))
        self.lb2 = lb2
        self.lb2 = self.lb2.to(device)

        # measurement points data
        path = 'pear.csv'
        in_data = pd.read_csv(path)
        in_data = in_data.values
        x_data = in_data[:, 0].reshape(-1, 1)
        y_data = in_data[:, 1].reshape(-1, 1)
        e11_data = 1 * in_data[:, 2].reshape(-1, 1)
        e12_data = 1 * in_data[:, 3].reshape(-1, 1)
        e22_data = 1 * in_data[:, 4].reshape(-1, 1)
        ### add noise ###
        # noise1 = np.random.normal(loc=0, scale=1.0, size=(49, 1))
        # e11_data = e11_data + noise1 * 0.01
        # noise2 = np.random.normal(loc=0, scale=1.0, size=(49, 1))
        # e12_data = e12_data + noise2 * 0.01
        # noise3 = np.random.normal(loc=0, scale=1.0, size=(49, 1))
        # e22_data = e22_data + noise3 * 0.01
        X_data = np.hstack((x_data, y_data))
        X_data = torch.tensor(X_data)
        self.X_data = X_data
        self.X_data = self.X_data.to(device)
        e11_data = torch.tensor(e11_data)
        self.e11_data = e11_data
        self.e11_data = self.e11_data.to(device)
        e12_data = torch.tensor(e12_data)
        self.e12_data = e12_data
        self.e12_data = self.e12_data.to(device)
        e22_data = torch.tensor(e22_data)
        self.e22_data = e22_data
        self.e22_data = self.e22_data.to(device)

        X_c = torch.tensor(X_c)
        self.X_c = X_c
        self.X_c = self.X_c.to(device)
        X_ub = torch.tensor(X_ub)
        self.X_ub = X_ub
        self.X_ub = self.X_ub.to(device)
        X_lb = torch.tensor(X_lb)
        self.X_lb = X_lb
        self.X_lb = self.X_lb.to(device)

        t = torch.tensor(t)
        self.t = t
        self.t = self.t.to(device)
        X_leftb = torch.tensor(X_leftb)
        self.X_leftb = X_leftb
        self.X_leftb = self.X_leftb.to(device)
        X_rightb = torch.tensor(X_rightb)
        self.X_rightb = X_rightb
        self.X_rightb = self.X_rightb.to(device)
        zerobd = torch.tensor(zerobd)
        self.zerobd = zerobd
        self.zerobd = self.zerobd.to(device)

        ### Geometric parameters ###
        a1 = np.random.rand() * 0.2 - 0.1
        b1 = np.random.rand() * 0.2 - 0.1
        self.lambda_1 = torch.tensor([a1], requires_grad=True).to(device)
        self.lambda_2 = torch.tensor([b1], requires_grad=True).to(device)
        self.lambda_3 = torch.tensor([0.05], requires_grad=True).to(device)
        self.lambda_1 = torch.nn.Parameter(self.lambda_1)
        self.lambda_2 = torch.nn.Parameter(self.lambda_2)
        self.lambda_3 = torch.nn.Parameter(self.lambda_3)

        self.model.register_parameter('lambda_1', self.lambda_1)
        self.model.register_parameter('lambda_2', self.lambda_2)
        self.model.register_parameter('lambda_3', self.lambda_3)

        a2 = np.random.rand() * 0.2 - 0.1
        b2 = np.random.rand() * 0.2 - 0.1
        self.lambda_4 = torch.tensor([a2], requires_grad=True).to(device)
        self.lambda_5 = torch.tensor([b2], requires_grad=True).to(device)
        self.lambda_6 = torch.tensor([0.05], requires_grad=True).to(device)
        self.lambda_4 = torch.nn.Parameter(self.lambda_4)
        self.lambda_5 = torch.nn.Parameter(self.lambda_5)
        self.lambda_6 = torch.nn.Parameter(self.lambda_6)

        self.model.register_parameter('lambda_4', self.lambda_4)
        self.model.register_parameter('lambda_5', self.lambda_5)
        self.model.register_parameter('lambda_6', self.lambda_6)

        a3 = np.random.rand() * 0.2 - 0.1
        b3 = np.random.rand() * 0.2 - 0.1
        self.lambda_7 = torch.tensor([a3], requires_grad=True).to(device)
        self.lambda_8 = torch.tensor([b3], requires_grad=True).to(device)
        self.lambda_9 = torch.tensor([0.05], requires_grad=True).to(device)
        self.lambda_7 = torch.nn.Parameter(self.lambda_7)
        self.lambda_8 = torch.nn.Parameter(self.lambda_8)
        self.lambda_9 = torch.nn.Parameter(self.lambda_9)

        self.model.register_parameter('lambda_7', self.lambda_7)
        self.model.register_parameter('lambda_8', self.lambda_8)
        self.model.register_parameter('lambda_9', self.lambda_9)

        a4 = np.random.rand() * 0.2 - 0.1
        b4 = np.random.rand() * 0.2 - 0.1
        self.lambda_10 = torch.tensor([a4], requires_grad=True).to(device)
        self.lambda_11 = torch.tensor([b4], requires_grad=True).to(device)
        self.lambda_12 = torch.tensor([0.05], requires_grad=True).to(device)
        self.lambda_10 = torch.nn.Parameter(self.lambda_10)
        self.lambda_11 = torch.nn.Parameter(self.lambda_11)
        self.lambda_12 = torch.nn.Parameter(self.lambda_12)

        self.model.register_parameter('lambda_10', self.lambda_10)
        self.model.register_parameter('lambda_11', self.lambda_11)
        self.model.register_parameter('lambda_12', self.lambda_12)

        a5 = np.random.rand() * 0.2 - 0.1
        b5 = np.random.rand() * 0.2 - 0.1
        self.lambda_13 = torch.tensor([a5], requires_grad=True).to(device)
        self.lambda_14 = torch.tensor([b5], requires_grad=True).to(device)
        self.lambda_15 = torch.tensor([0.05], requires_grad=True).to(device)
        self.lambda_13 = torch.nn.Parameter(self.lambda_13)
        self.lambda_14 = torch.nn.Parameter(self.lambda_14)
        self.lambda_15 = torch.nn.Parameter(self.lambda_15)

        self.model.register_parameter('lambda_13', self.lambda_13)
        self.model.register_parameter('lambda_14', self.lambda_14)
        self.model.register_parameter('lambda_15', self.lambda_15)

        a6 = np.random.rand() * 0.2 - 0.1
        b6 = np.random.rand() * 0.2 - 0.1
        self.lambda_16 = torch.tensor([a6], requires_grad=True).to(device)
        self.lambda_17 = torch.tensor([b6], requires_grad=True).to(device)
        self.lambda_18 = torch.tensor([0.05], requires_grad=True).to(device)
        self.lambda_16 = torch.nn.Parameter(self.lambda_16)
        self.lambda_17 = torch.nn.Parameter(self.lambda_17)
        self.lambda_18 = torch.nn.Parameter(self.lambda_18)

        self.model.register_parameter('lambda_16', self.lambda_16)
        self.model.register_parameter('lambda_17', self.lambda_17)
        self.model.register_parameter('lambda_18', self.lambda_18)

        a7 = np.random.rand() * 0.2 - 0.1
        b7 = np.random.rand() * 0.2 - 0.1
        self.lambda_19 = torch.tensor([a7], requires_grad=True).to(device)
        self.lambda_20 = torch.tensor([b7], requires_grad=True).to(device)
        self.lambda_21 = torch.tensor([0.05], requires_grad=True).to(device)
        self.lambda_19 = torch.nn.Parameter(self.lambda_19)
        self.lambda_20 = torch.nn.Parameter(self.lambda_20)
        self.lambda_21 = torch.nn.Parameter(self.lambda_21)

        self.model.register_parameter('lambda_19', self.lambda_19)
        self.model.register_parameter('lambda_20', self.lambda_20)
        self.model.register_parameter('lambda_21', self.lambda_21)

        a8 = np.random.rand() * 0.2 - 0.1
        b8 = np.random.rand() * 0.2 - 0.1
        self.lambda_22 = torch.tensor([a8], requires_grad=True).to(device)
        self.lambda_23 = torch.tensor([b8], requires_grad=True).to(device)
        self.lambda_24 = torch.tensor([0.05], requires_grad=True).to(device)
        self.lambda_22 = torch.nn.Parameter(self.lambda_22)
        self.lambda_23 = torch.nn.Parameter(self.lambda_23)
        self.lambda_24 = torch.nn.Parameter(self.lambda_24)

        self.model.register_parameter('lambda_22', self.lambda_22)
        self.model.register_parameter('lambda_23', self.lambda_23)
        self.model.register_parameter('lambda_24', self.lambda_24)

        a9 = np.random.rand() * 0.2 - 0.1
        b9 = np.random.rand() * 0.2 - 0.1
        self.lambda_25 = torch.tensor([a9], requires_grad=True).to(device)
        self.lambda_26 = torch.tensor([b9], requires_grad=True).to(device)
        self.lambda_27 = torch.tensor([0.05], requires_grad=True).to(device)
        self.lambda_25 = torch.nn.Parameter(self.lambda_25)
        self.lambda_26 = torch.nn.Parameter(self.lambda_26)
        self.lambda_27 = torch.nn.Parameter(self.lambda_27)

        self.model.register_parameter('lambda_25', self.lambda_25)
        self.model.register_parameter('lambda_26', self.lambda_26)
        self.model.register_parameter('lambda_27', self.lambda_27)

        a10 = np.random.rand() * 0.2 - 0.1
        b10 = np.random.rand() * 0.2 - 0.1
        self.lambda_28 = torch.tensor([a10], requires_grad=True).to(device)
        self.lambda_29 = torch.tensor([b10], requires_grad=True).to(device)
        self.lambda_30 = torch.tensor([0.05], requires_grad=True).to(device)
        self.lambda_28 = torch.nn.Parameter(self.lambda_28)
        self.lambda_29 = torch.nn.Parameter(self.lambda_29)
        self.lambda_30 = torch.nn.Parameter(self.lambda_30)

        self.model.register_parameter('lambda_28', self.lambda_28)
        self.model.register_parameter('lambda_29', self.lambda_29)
        self.model.register_parameter('lambda_30', self.lambda_30)

        a11 = np.random.rand() * 0.2 - 0.1
        b11 = np.random.rand() * 0.2 - 0.1
        self.lambda_31 = torch.tensor([a11], requires_grad=True).to(device)
        self.lambda_32 = torch.tensor([b11], requires_grad=True).to(device)
        self.lambda_33 = torch.tensor([0.05], requires_grad=True).to(device)
        self.lambda_31 = torch.nn.Parameter(self.lambda_31)
        self.lambda_32 = torch.nn.Parameter(self.lambda_32)
        self.lambda_33 = torch.nn.Parameter(self.lambda_33)

        self.model.register_parameter('lambda_31', self.lambda_31)
        self.model.register_parameter('lambda_32', self.lambda_32)
        self.model.register_parameter('lambda_33', self.lambda_33)

        a12 = np.random.rand() * 0.2 - 0.1
        b12 = np.random.rand() * 0.2 - 0.1
        self.lambda_34 = torch.tensor([a12], requires_grad=True).to(device)
        self.lambda_35 = torch.tensor([b12], requires_grad=True).to(device)
        self.lambda_36 = torch.tensor([0.05], requires_grad=True).to(device)
        self.lambda_34 = torch.nn.Parameter(self.lambda_34)
        self.lambda_35 = torch.nn.Parameter(self.lambda_35)
        self.lambda_36 = torch.nn.Parameter(self.lambda_36)

        self.model.register_parameter('lambda_34', self.lambda_34)
        self.model.register_parameter('lambda_35', self.lambda_35)
        self.model.register_parameter('lambda_36', self.lambda_36)

        a13 = np.random.rand() * 0.2 - 0.1
        b13 = np.random.rand() * 0.2 - 0.1
        self.lambda_37 = torch.tensor([a13], requires_grad=True).to(device)
        self.lambda_38 = torch.tensor([b13], requires_grad=True).to(device)
        self.lambda_39 = torch.tensor([0.05], requires_grad=True).to(device)
        self.lambda_37 = torch.nn.Parameter(self.lambda_37)
        self.lambda_38 = torch.nn.Parameter(self.lambda_38)
        self.lambda_39 = torch.nn.Parameter(self.lambda_39)

        self.model.register_parameter('lambda_37', self.lambda_37)
        self.model.register_parameter('lambda_38', self.lambda_38)
        self.model.register_parameter('lambda_39', self.lambda_39)

        a14 = np.random.rand() * 0.2 - 0.1
        b14 = np.random.rand() * 0.2 - 0.1
        self.lambda_40 = torch.tensor([a14], requires_grad=True).to(device)
        self.lambda_41 = torch.tensor([b14], requires_grad=True).to(device)
        self.lambda_42 = torch.tensor([0.05], requires_grad=True).to(device)
        self.lambda_40 = torch.nn.Parameter(self.lambda_40)
        self.lambda_41 = torch.nn.Parameter(self.lambda_41)
        self.lambda_42 = torch.nn.Parameter(self.lambda_42)

        self.model.register_parameter('lambda_40', self.lambda_40)
        self.model.register_parameter('lambda_41', self.lambda_41)
        self.model.register_parameter('lambda_42', self.lambda_42)

        a15 = np.random.rand() * 0.2 - 0.1
        b15 = np.random.rand() * 0.2 - 0.1
        self.lambda_43 = torch.tensor([a15], requires_grad=True).to(device)
        self.lambda_44 = torch.tensor([b15], requires_grad=True).to(device)
        self.lambda_45 = torch.tensor([0.05], requires_grad=True).to(device)
        self.lambda_43 = torch.nn.Parameter(self.lambda_43)
        self.lambda_44 = torch.nn.Parameter(self.lambda_44)
        self.lambda_45 = torch.nn.Parameter(self.lambda_45)

        self.model.register_parameter('lambda_43', self.lambda_43)
        self.model.register_parameter('lambda_44', self.lambda_44)
        self.model.register_parameter('lambda_45', self.lambda_45)

        a16 = np.random.rand() * 0.2 - 0.1
        b16 = np.random.rand() * 0.2 - 0.1
        self.lambda_46 = torch.tensor([a16], requires_grad=True).to(device)
        self.lambda_47 = torch.tensor([b16], requires_grad=True).to(device)
        self.lambda_48 = torch.tensor([0.05], requires_grad=True).to(device)
        self.lambda_46 = torch.nn.Parameter(self.lambda_46)
        self.lambda_47 = torch.nn.Parameter(self.lambda_47)
        self.lambda_48 = torch.nn.Parameter(self.lambda_48)

        self.model.register_parameter('lambda_46', self.lambda_46)
        self.model.register_parameter('lambda_47', self.lambda_47)
        self.model.register_parameter('lambda_48', self.lambda_48)

        self.X_c.requires_grad = True

        self.loss_pre = 10
        # optimizer
        self.criterion = torch.nn.MSELoss()
        self.iter = 0
        self.optimizer = torch.optim.LBFGS(
            self.model.parameters(),
            lr=1,
            max_iter=98000,
            max_eval=98000,
            history_size=50,
            tolerance_grad=1e-7,
            tolerance_change=1.0 * np.finfo(float).eps
        )
        self.optimizer2 = torch.optim.LBFGS(
            self.model.parameters(),
            lr=1,
            max_iter=2000,
            max_eval=2000,
            history_size=50,
            tolerance_grad=1e-7,
            tolerance_change=1.0 * np.finfo(float).eps
        )
        self.optimizer3 = torch.optim.LBFGS(
            self.model.parameters(),
            lr=1,
            max_iter=20000,
            max_eval=20000,
            history_size=50,
            tolerance_grad=1e-7,
            tolerance_change=1.0 * np.finfo(float).eps
        )
        self.adam = torch.optim.Adam(self.model.parameters(), lr=0.0001)

    ### Loss Function ###
    def loss_func(self):
        self.optimizer.zero_grad()

        Y_ub = self.model(self.X_ub)
        s12_ub = Y_ub[:, 3].reshape(-1, 1)
        s22_ub = Y_ub[:, 4].reshape(-1, 1)
        phi_ub = Y_ub[:, 5].reshape(-1, 1)
        loss_ub1 = self.criterion(s22_ub, self.t)
        loss_ub2 = self.criterion(s12_ub, self.zerobd)
        loss_phi1 = self.criterion(phi_ub, self.zerobd)

        Y_lb = self.model(self.X_lb)
        v_lb = Y_lb[:, 1].reshape(-1, 1)
        s12_lb = Y_lb[:, 3].reshape(-1, 1)
        phi_lb = Y_lb[:, 5].reshape(-1, 1)
        loss_lb1 = self.criterion(v_lb, self.zerobd)
        loss_lb2 = self.criterion(s12_lb, self.zerobd)
        loss_phi2 = self.criterion(phi_lb, self.zerobd)

        Y_lb2 = self.model(self.X_lb2)
        u_lb2 = Y_lb2[:, 0]
        v_lb2 = Y_lb2[:, 1]
        loss_lb3 = self.criterion(u_lb2, self.lb2) + self.criterion(v_lb2, self.lb2)

        Y_left = self.model(self.X_leftb)
        s11_left = Y_left[:, 2].reshape(-1, 1)
        s12_left = Y_left[:, 3].reshape(-1, 1)
        phi_left = Y_left[:, 5].reshape(-1, 1)
        loss_left = self.criterion(s11_left, self.zerobd) + self.criterion(s12_left, self.zerobd)
        loss_phi3 = self.criterion(phi_left, self.zerobd)

        Y_right = self.model(self.X_rightb)
        s11_right = Y_right[:, 2].reshape(-1, 1)
        s12_right = Y_right[:, 3].reshape(-1, 1)
        phi_right = Y_right[:, 5].reshape(-1, 1)
        loss_right = self.criterion(s11_right, self.zerobd) + self.criterion(s12_right, self.zerobd)
        loss_phi4 = self.criterion(phi_right, self.zerobd)

        Y_in = self.model(self.X_data)
        s11_in = Y_in[:, 2].reshape(-1, 1)
        s12_in = Y_in[:, 3].reshape(-1, 1)
        s22_in = Y_in[:, 4].reshape(-1, 1)
        s11_data = (E / (1 - miu ** 2)) * (self.e11_data + miu * self.e22_data)
        s22_data = (E / (1 - miu ** 2)) * (self.e22_data + miu * self.e11_data)
        s12_data = (E / (1 + miu)) * self.e12_data
        loss_data = self.criterion(s11_in, s11_data) + self.criterion(s12_in, s12_data) + self.criterion(s22_in,
                                                                                                         s22_data)

        Y_c = self.model(self.X_c)
        u = Y_c[:, 0].reshape(-1, 1)
        v = Y_c[:, 1].reshape(-1, 1)
        s11 = Y_c[:, 2]
        s12 = Y_c[:, 3]
        s22 = Y_c[:, 4]
        phi = Y_c[:, 5]
        # phi = torch.clamp(phi, 0, 1)
        phi_c = (-torch.tanh(
            (torch.sqrt(((self.X_c[:, 0]) ** 2 + (self.X_c[:, 1]) ** 2)) - 0.1) / (
                    math.sqrt(2) * 0.005)) + 1) * 0.5
        # phi_star = torch.zeros_like(phi).to(device)
        loss_phi = self.criterion(phi, phi_c)

        du_dX = torch.autograd.grad(inputs=self.X_c, outputs=u, grad_outputs=torch.ones_like(u), retain_graph=True,
                                    create_graph=True)[0]
        dv_dX = torch.autograd.grad(inputs=self.X_c, outputs=v, grad_outputs=torch.ones_like(v), retain_graph=True,
                                    create_graph=True)[0]
        du_dx = du_dX[:, 0]
        du_dy = du_dX[:, 1]
        dv_dx = dv_dX[:, 0]
        dv_dy = dv_dX[:, 1]

        e11 = du_dx
        e22 = dv_dy
        e12 = 0.5 * (du_dy + dv_dx)

        s11_p = (E / (1 - miu ** 2)) * (e11 + miu * e22)
        s22_p = (E / (1 - miu ** 2)) * (e22 + miu * e11)
        s12_p = (E / (1 + miu)) * e12

        loss_f1 = self.criterion(s11, s11_p)
        loss_f2 = self.criterion(s12, s12_p)
        loss_f3 = self.criterion(s22, s22_p)

        ds11_dx = \
            torch.autograd.grad(inputs=self.X_c, outputs=s11, grad_outputs=torch.ones_like(s11), retain_graph=True,
                                create_graph=True)[0][:, 0]
        ds12_dx = \
            torch.autograd.grad(inputs=self.X_c, outputs=s12, grad_outputs=torch.ones_like(s12), retain_graph=True,
                                create_graph=True)[0][:, 0]
        ds12_dy = \
            torch.autograd.grad(inputs=self.X_c, outputs=s12, grad_outputs=torch.ones_like(s12), retain_graph=True,
                                create_graph=True)[0][:, 1]
        ds22_dy = \
            torch.autograd.grad(inputs=self.X_c, outputs=s22, grad_outputs=torch.ones_like(s22), retain_graph=True,
                                create_graph=True)[0][:, 1]
        g_phi = (1 - phi) ** 2

        dphi_dx = \
            torch.autograd.grad(inputs=self.X_c, outputs=phi, grad_outputs=torch.ones_like(phi), retain_graph=True,
                                create_graph=True)[0][:, 0]
        dphi_dy = \
            torch.autograd.grad(inputs=self.X_c, outputs=phi, grad_outputs=torch.ones_like(phi), retain_graph=True,
                                create_graph=True)[0][:, 1]
        dg_dx = 2 * (phi - 1) * dphi_dx
        dg_dy = 2 * (phi - 1) * dphi_dy
        loss_pde1 = self.criterion(dg_dx * s11 + dg_dy * s12, -ds11_dx * g_phi - ds12_dy * g_phi)
        loss_pde2 = self.criterion(dg_dx * s12 + dg_dy * s22, -ds12_dx * g_phi - ds22_dy * g_phi)
        loss_pde = loss_pde1 + loss_pde2
        loss_bc = loss_ub1 + (loss_lb1 + loss_lb2 + loss_lb3 + loss_ub2 + loss_left + loss_right)
        loss_f = loss_f1 + loss_f2 + loss_f3
        loss_phi_bc = loss_phi1 + loss_phi2 + loss_phi3 + loss_phi4

        loss = 1 * loss_ub1 + 10 * (
                1 * loss_lb1 + loss_lb2 + loss_ub2 + loss_lb3 + loss_left + loss_right) + 2 * loss_f + loss_pde + 10 * loss_data + loss_phi_bc
        loss.backward()

        ### plot ###
        if self.iter % 1000 == 0:
            fig, ax = plt.subplots()
            ax.set_aspect('equal')
            X_c_current = self.X_c.cpu()
            X_c_current = X_c_current.detach().numpy()
            phi_np = phi.cpu().detach().numpy()
            pm = ax.scatter(X_c_current[:, 0], X_c_current[:, 1], c=phi_np, cmap='rainbow', marker='o', s=2, alpha=1,
                            edgecolors='none')
            cbar = fig.colorbar(pm, ax=ax)
            ax.set_title('Iter:{}'.format(self.iter), fontsize=15)
            plt.pause(0.5)
            plt.savefig('fig/Pear_{}.png'.format(self.iter))
            plt.close()

        if self.iter % 100 == 0:
            dcrease = -((loss.item() - self.loss_pre) / self.loss_pre) * 100
            self.loss_pre = loss.item()
            print(self.iter, loss.item(), f"dcrease{dcrease:.2f}%", 'CP Point:', self.X_c.size(0))
            print(f"loss_BC1:{loss_ub1.item()}")
            print(
                f"loss_BC2:{1 * loss_lb1 + loss_left + loss_right}")
            print(f"loss_f:{loss_f.item()}")
            print(f"loss_pde:{loss_pde1.item() + loss_pde2.item()}")
            print(f"loss_phi_bc:{loss_phi1.item() + loss_phi2.item() + loss_phi3.item() + loss_phi4.item()}")
            print(f"loss_phi:{loss_phi.item()}")
            print(f"loss_data:{loss_data.item()}")

        if self.iter % 10000 == 0:
            torch.save(net, 'Pear.pth')

        writer.add_scalar('loss', loss.item(), self.iter)
        writer.add_scalar('loss_pde', loss_pde.item(), self.iter)
        writer.add_scalar('loss_f', loss_f.item(), self.iter)
        writer.add_scalar('loss_bc', loss_bc.item(), self.iter)
        writer.add_scalar('loss_phi', loss_phi.item(), self.iter)
        writer.add_scalar('loss_data', loss_data.item(), self.iter)

        self.iter = self.iter + 1

        return loss

    def loss_func2(self):
        self.optimizer.zero_grad()

        Y_ub = self.model(self.X_ub)
        s12_ub = Y_ub[:, 3].reshape(-1, 1)
        s22_ub = Y_ub[:, 4].reshape(-1, 1)
        phi_ub = Y_ub[:, 5].reshape(-1, 1)
        loss_ub1 = self.criterion(s22_ub, self.t)
        loss_ub2 = self.criterion(s12_ub, self.zerobd)
        loss_phi1 = self.criterion(phi_ub, self.zerobd)

        Y_lb = self.model(self.X_lb)
        v_lb = Y_lb[:, 1].reshape(-1, 1)
        s12_lb = Y_lb[:, 3].reshape(-1, 1)
        phi_lb = Y_lb[:, 5].reshape(-1, 1)
        loss_lb1 = self.criterion(v_lb, self.zerobd)
        loss_lb2 = self.criterion(s12_lb, self.zerobd)

        loss_phi2 = self.criterion(phi_lb, self.zerobd)

        Y_lb2 = self.model(self.X_lb2)
        u_lb2 = Y_lb2[:, 0]
        v_lb2 = Y_lb2[:, 1]
        loss_lb3 = self.criterion(u_lb2, self.lb2) + self.criterion(v_lb2, self.lb2)

        Y_left = self.model(self.X_leftb)
        s11_left = Y_left[:, 2].reshape(-1, 1)
        s12_left = Y_left[:, 3].reshape(-1, 1)
        phi_left = Y_left[:, 5].reshape(-1, 1)
        loss_left = self.criterion(s11_left, self.zerobd) + self.criterion(s12_left, self.zerobd)
        loss_phi3 = self.criterion(phi_left, self.zerobd)

        Y_right = self.model(self.X_rightb)
        s11_right = Y_right[:, 2].reshape(-1, 1)
        s12_right = Y_right[:, 3].reshape(-1, 1)
        phi_right = Y_right[:, 5].reshape(-1, 1)
        loss_right = self.criterion(s11_right, self.zerobd) + self.criterion(s12_right, self.zerobd)
        loss_phi4 = self.criterion(phi_right, self.zerobd)

        Y_in = self.model(self.X_data)
        s11_in = Y_in[:, 2].reshape(-1, 1)
        s12_in = Y_in[:, 3].reshape(-1, 1)
        s22_in = Y_in[:, 4].reshape(-1, 1)
        s11_data = (E / (1 - miu ** 2)) * (self.e11_data + miu * self.e22_data)
        s22_data = (E / (1 - miu ** 2)) * (self.e22_data + miu * self.e11_data)
        s12_data = (E / (1 + miu)) * self.e12_data
        loss_data = self.criterion(s11_in, s11_data) + self.criterion(s12_in, s12_data) + self.criterion(s22_in,
                                                                                                         s22_data)

        Y_c = self.model(self.X_c)
        u = Y_c[:, 0].reshape(-1, 1)
        v = Y_c[:, 1].reshape(-1, 1)
        s11 = Y_c[:, 2]
        s12 = Y_c[:, 3]
        s22 = Y_c[:, 4]
        phi = Y_c[:, 5]
        x_c1 = self.lambda_1
        y_c1 = self.lambda_2
        r1 = torch.sqrt(self.lambda_3 ** 2)
        x_c2 = self.lambda_4
        y_c2 = self.lambda_5
        r2 = torch.sqrt(self.lambda_6 ** 2)
        x_c3 = self.lambda_7
        y_c3 = self.lambda_8
        r3 = torch.sqrt(self.lambda_9 ** 2)
        x_c4 = self.lambda_10
        y_c4 = self.lambda_11
        r4 = torch.sqrt(self.lambda_12 ** 2)
        x_c5 = self.lambda_13
        y_c5 = self.lambda_14
        r5 = torch.sqrt(self.lambda_15 ** 2)
        x_c6 = self.lambda_16
        y_c6 = self.lambda_17
        r6 = torch.sqrt(self.lambda_18 ** 2)
        x_c7 = self.lambda_19
        y_c7 = self.lambda_20
        r7 = torch.sqrt(self.lambda_21 ** 2)
        x_c8 = self.lambda_22
        y_c8 = self.lambda_23
        r8 = torch.sqrt(self.lambda_24 ** 2)
        x_c9 = self.lambda_25
        y_c9 = self.lambda_26
        r9 = torch.sqrt(self.lambda_27 ** 2)
        x_c10 = self.lambda_28
        y_c10 = self.lambda_29
        r10 = torch.sqrt(self.lambda_30 ** 2)
        x_c11 = self.lambda_31
        y_c11 = self.lambda_32
        r11 = torch.sqrt(self.lambda_33 ** 2)
        x_c12 = self.lambda_34
        y_c12 = self.lambda_35
        r12 = torch.sqrt(self.lambda_36 ** 2)
        x_c13 = self.lambda_37
        y_c13 = self.lambda_38
        r13 = torch.sqrt(self.lambda_39 ** 2)
        x_c14 = self.lambda_40
        y_c14 = self.lambda_41
        r14 = torch.sqrt(self.lambda_42 ** 2)
        x_c15 = self.lambda_43
        y_c15 = self.lambda_44
        r15 = torch.sqrt(self.lambda_45 ** 2)
        x_c16 = self.lambda_46
        y_c16 = self.lambda_47
        r16 = torch.sqrt(self.lambda_48 ** 2)
        phi_c = (-torch.tanh(
            (torch.sqrt(((self.X_c[:, 0] - x_c1) ** 2 + (self.X_c[:, 1] - y_c1) ** 2)) - r1) / (
                    math.sqrt(2) * 0.005)) + 1) * 0.5 + (-torch.tanh(
            (torch.sqrt(((self.X_c[:, 0] - x_c2) ** 2 + (self.X_c[:, 1] - y_c2) ** 2)) - r2) / (
                    math.sqrt(2) * 0.005)) + 1) * 0.5 + (-torch.tanh(
            (torch.sqrt(((self.X_c[:, 0] - x_c3) ** 2 + (self.X_c[:, 1] - y_c3) ** 2)) - r3) / (
                    math.sqrt(2) * 0.005)) + 1) * 0.5 + (-torch.tanh(
            (torch.sqrt(((self.X_c[:, 0] - x_c4) ** 2 + (self.X_c[:, 1] - y_c4) ** 2)) - r4) / (
                    math.sqrt(2) * 0.005)) + 1) * 0.5 + (-torch.tanh(
            (torch.sqrt(((self.X_c[:, 0] - x_c5) ** 2 + (self.X_c[:, 1] - y_c5) ** 2)) - r5) / (
                    math.sqrt(2) * 0.005)) + 1) * 0.5 + (-torch.tanh(
            (torch.sqrt(((self.X_c[:, 0] - x_c6) ** 2 + (self.X_c[:, 1] - y_c6) ** 2)) - r6) / (
                    math.sqrt(2) * 0.005)) + 1) * 0.5 + (-torch.tanh(
            (torch.sqrt(((self.X_c[:, 0] - x_c7) ** 2 + (self.X_c[:, 1] - y_c7) ** 2)) - r7) / (
                    math.sqrt(2) * 0.005)) + 1) * 0.5 + (-torch.tanh(
            (torch.sqrt(((self.X_c[:, 0] - x_c8) ** 2 + (self.X_c[:, 1] - y_c8) ** 2)) - r8) / (
                    math.sqrt(2) * 0.005)) + 1) * 0.5 + (-torch.tanh(
            (torch.sqrt(((self.X_c[:, 0] - x_c9) ** 2 + (self.X_c[:, 1] - y_c9) ** 2)) - r9) / (
                    math.sqrt(2) * 0.005)) + 1) * 0.5 + (-torch.tanh(
            (torch.sqrt(((self.X_c[:, 0] - x_c10) ** 2 + (self.X_c[:, 1] - y_c10) ** 2)) - r10) / (
                    math.sqrt(2) * 0.005)) + 1) * 0.5 + (-torch.tanh(
            (torch.sqrt(((self.X_c[:, 0] - x_c11) ** 2 + (self.X_c[:, 1] - y_c11) ** 2)) - r11) / (
                    math.sqrt(2) * 0.005)) + 1) * 0.5 + (-torch.tanh(
            (torch.sqrt(((self.X_c[:, 0] - x_c12) ** 2 + (self.X_c[:, 1] - y_c12) ** 2)) - r12) / (
                    math.sqrt(2) * 0.005)) + 1) * 0.5 + (-torch.tanh(
            (torch.sqrt(((self.X_c[:, 0] - x_c13) ** 2 + (self.X_c[:, 1] - y_c13) ** 2)) - r13) / (
                    math.sqrt(2) * 0.005)) + 1) * 0.5 + (-torch.tanh(
            (torch.sqrt(((self.X_c[:, 0] - x_c14) ** 2 + (self.X_c[:, 1] - y_c14) ** 2)) - r14) / (
                    math.sqrt(2) * 0.005)) + 1) * 0.5 + (-torch.tanh(
            (torch.sqrt(((self.X_c[:, 0] - x_c15) ** 2 + (self.X_c[:, 1] - y_c15) ** 2)) - r15) / (
                    math.sqrt(2) * 0.005)) + 1) * 0.5 + (-torch.tanh(
            (torch.sqrt(((self.X_c[:, 0] - x_c16) ** 2 + (self.X_c[:, 1] - y_c16) ** 2)) - r16) / (
                    math.sqrt(2) * 0.005)) + 1) * 0.5
        phi_c = torch.clamp(phi_c, 0, 1)
        loss_phi = self.criterion(phi, phi_c)

        du_dX = torch.autograd.grad(inputs=self.X_c, outputs=u, grad_outputs=torch.ones_like(u), retain_graph=True,
                                    create_graph=True)[0]
        dv_dX = torch.autograd.grad(inputs=self.X_c, outputs=v, grad_outputs=torch.ones_like(v), retain_graph=True,
                                    create_graph=True)[0]
        du_dx = du_dX[:, 0]
        du_dy = du_dX[:, 1]
        dv_dx = dv_dX[:, 0]
        dv_dy = dv_dX[:, 1]

        e11 = du_dx
        e22 = dv_dy
        e12 = 0.5 * (du_dy + dv_dx)

        s11_p = (E / (1 - miu ** 2)) * (e11 + miu * e22)
        s22_p = (E / (1 - miu ** 2)) * (e22 + miu * e11)
        s12_p = (E / (1 + miu)) * e12

        loss_f1 = self.criterion(s11, s11_p)
        loss_f2 = self.criterion(s12, s12_p)
        loss_f3 = self.criterion(s22, s22_p)

        ds11_dx = \
            torch.autograd.grad(inputs=self.X_c, outputs=s11, grad_outputs=torch.ones_like(s11), retain_graph=True,
                                create_graph=True)[0][:, 0]
        ds12_dx = \
            torch.autograd.grad(inputs=self.X_c, outputs=s12, grad_outputs=torch.ones_like(s12), retain_graph=True,
                                create_graph=True)[0][:, 0]
        ds12_dy = \
            torch.autograd.grad(inputs=self.X_c, outputs=s12, grad_outputs=torch.ones_like(s12), retain_graph=True,
                                create_graph=True)[0][:, 1]
        ds22_dy = \
            torch.autograd.grad(inputs=self.X_c, outputs=s22, grad_outputs=torch.ones_like(s22), retain_graph=True,
                                create_graph=True)[0][:, 1]
        g_phi = (1 - phi) ** 2

        dphi_dx = \
            torch.autograd.grad(inputs=self.X_c, outputs=phi, grad_outputs=torch.ones_like(phi), retain_graph=True,
                                create_graph=True)[0][:, 0]
        dphi_dy = \
            torch.autograd.grad(inputs=self.X_c, outputs=phi, grad_outputs=torch.ones_like(phi), retain_graph=True,
                                create_graph=True)[0][:, 1]
        dg_dx = 2 * (phi - 1) * dphi_dx
        dg_dy = 2 * (phi - 1) * dphi_dy
        loss_pde1 = self.criterion(dg_dx * s11 + dg_dy * s12, -ds11_dx * g_phi - ds12_dy * g_phi)
        loss_pde2 = self.criterion(dg_dx * s12 + dg_dy * s22, -ds12_dx * g_phi - ds22_dy * g_phi)
        loss_pde = loss_pde1 + loss_pde2
        loss_bc = 1 * loss_ub1 + (1 * loss_lb1 + loss_lb2 + loss_lb3 + loss_ub2 + loss_left + loss_right)
        loss_f = loss_f1 + loss_f2 + loss_f3
        loss_phi_bc = loss_phi1 + loss_phi2 + loss_phi3 + loss_phi4

        loss = 1 * loss_ub1 + 10 * (
                1 * loss_lb1 + loss_lb2 + loss_lb3 + loss_ub2 + loss_left + loss_right) + 2 * loss_f + loss_pde + 10 * loss_data + loss_phi_bc + 10 * loss_phi
        loss.backward()

        if self.iter % 100 == 0:
            dcrease = -((loss.item() - self.loss_pre) / self.loss_pre) * 100
            self.loss_pre = loss.item()
            print(self.iter, loss.item(), f"dcrease{dcrease:.2f}%", 'CP Point:', self.X_c.size(0))
            print(f"loss_BC1:{loss_ub1.item()}")
            print(
                f"loss_BC2:{1 * loss_lb1 + loss_left + loss_right}")
            print(f"loss_f:{loss_f.item()}")
            print(f"loss_pde:{loss_pde1.item() + loss_pde2.item()}")
            print(f"loss_phi_bc:{loss_phi1.item() + loss_phi2.item() + loss_phi3.item() + loss_phi4.item()}")
            print(f"loss_phi:{loss_phi.item()}")
            print(f"loss_data:{loss_data.item()}")

        ### plot ###
        if self.iter % 1000 == 0:
            fig, ax = plt.subplots()
            ax.set_aspect('equal')
            X_c_current = self.X_c.cpu()
            X_c_current = X_c_current.detach().numpy()
            phi_np = phi.cpu().detach().numpy()
            pm = ax.scatter(X_c_current[:, 0], X_c_current[:, 1], c=phi_np, cmap='rainbow', marker='o', s=2, alpha=1,
                            edgecolors='none')
            cbar = fig.colorbar(pm, ax=ax)
            ax.set_title('Iter:{}'.format(self.iter), fontsize=15)
            plt.pause(0.5)
            plt.savefig('fig/Pear_{}.png'.format(self.iter))
            plt.close()

        if self.iter % 10000 == 0:
            torch.save(net, 'Pear.pth')

        writer.add_scalar('loss', loss.item(), self.iter)
        writer.add_scalar('loss_pde', loss_pde.item(), self.iter)
        writer.add_scalar('loss_f', loss_f.item(), self.iter)
        writer.add_scalar('loss_bc', loss_bc.item(), self.iter)
        writer.add_scalar('loss_phi', loss_phi.item(), self.iter)
        writer.add_scalar('loss_data', loss_data.item(), self.iter)

        self.iter = self.iter + 1

        return loss

    def loss_func3(self):
        self.optimizer.zero_grad()
        Y_c = self.model(self.X_c)
        phi = Y_c[:, 5]
        x_c1 = self.lambda_1
        y_c1 = self.lambda_2
        r1 = torch.sqrt(self.lambda_3 ** 2)
        x_c2 = self.lambda_4
        y_c2 = self.lambda_5
        r2 = torch.sqrt(self.lambda_6 ** 2)
        x_c3 = self.lambda_7
        y_c3 = self.lambda_8
        r3 = torch.sqrt(self.lambda_9 ** 2)
        x_c4 = self.lambda_10
        y_c4 = self.lambda_11
        r4 = torch.sqrt(self.lambda_12 ** 2)
        x_c5 = self.lambda_13
        y_c5 = self.lambda_14
        r5 = torch.sqrt(self.lambda_15 ** 2)
        x_c6 = self.lambda_16
        y_c6 = self.lambda_17
        r6 = torch.sqrt(self.lambda_18 ** 2)
        x_c7 = self.lambda_19
        y_c7 = self.lambda_20
        r7 = torch.sqrt(self.lambda_21 ** 2)
        x_c8 = self.lambda_22
        y_c8 = self.lambda_23
        r8 = torch.sqrt(self.lambda_24 ** 2)
        x_c9 = self.lambda_25
        y_c9 = self.lambda_26
        r9 = torch.sqrt(self.lambda_27 ** 2)
        x_c10 = self.lambda_28
        y_c10 = self.lambda_29
        r10 = torch.sqrt(self.lambda_30 ** 2)
        x_c11 = self.lambda_31
        y_c11 = self.lambda_32
        r11 = torch.sqrt(self.lambda_33 ** 2)
        x_c12 = self.lambda_34
        y_c12 = self.lambda_35
        r12 = torch.sqrt(self.lambda_36 ** 2)
        x_c13 = self.lambda_37
        y_c13 = self.lambda_38
        r13 = torch.sqrt(self.lambda_39 ** 2)
        x_c14 = self.lambda_40
        y_c14 = self.lambda_41
        r14 = torch.sqrt(self.lambda_42 ** 2)
        x_c15 = self.lambda_43
        y_c15 = self.lambda_44
        r15 = torch.sqrt(self.lambda_45 ** 2)
        x_c16 = self.lambda_46
        y_c16 = self.lambda_47
        r16 = torch.sqrt(self.lambda_48 ** 2)
        phi_c = (-torch.tanh(
            (torch.sqrt(((self.X_c[:, 0] - x_c1) ** 2 + (self.X_c[:, 1] - y_c1) ** 2)) - r1) / (
                    math.sqrt(2) * 0.005)) + 1) * 0.5 + (-torch.tanh(
            (torch.sqrt(((self.X_c[:, 0] - x_c2) ** 2 + (self.X_c[:, 1] - y_c2) ** 2)) - r2) / (
                    math.sqrt(2) * 0.005)) + 1) * 0.5 + (-torch.tanh(
            (torch.sqrt(((self.X_c[:, 0] - x_c3) ** 2 + (self.X_c[:, 1] - y_c3) ** 2)) - r3) / (
                    math.sqrt(2) * 0.005)) + 1) * 0.5 + (-torch.tanh(
            (torch.sqrt(((self.X_c[:, 0] - x_c4) ** 2 + (self.X_c[:, 1] - y_c4) ** 2)) - r4) / (
                    math.sqrt(2) * 0.005)) + 1) * 0.5 + (-torch.tanh(
            (torch.sqrt(((self.X_c[:, 0] - x_c5) ** 2 + (self.X_c[:, 1] - y_c5) ** 2)) - r5) / (
                    math.sqrt(2) * 0.005)) + 1) * 0.5 + (-torch.tanh(
            (torch.sqrt(((self.X_c[:, 0] - x_c6) ** 2 + (self.X_c[:, 1] - y_c6) ** 2)) - r6) / (
                    math.sqrt(2) * 0.005)) + 1) * 0.5 + (-torch.tanh(
            (torch.sqrt(((self.X_c[:, 0] - x_c7) ** 2 + (self.X_c[:, 1] - y_c7) ** 2)) - r7) / (
                    math.sqrt(2) * 0.005)) + 1) * 0.5 + (-torch.tanh(
            (torch.sqrt(((self.X_c[:, 0] - x_c8) ** 2 + (self.X_c[:, 1] - y_c8) ** 2)) - r8) / (
                    math.sqrt(2) * 0.005)) + 1) * 0.5 + (-torch.tanh(
            (torch.sqrt(((self.X_c[:, 0] - x_c9) ** 2 + (self.X_c[:, 1] - y_c9) ** 2)) - r9) / (
                    math.sqrt(2) * 0.005)) + 1) * 0.5 + (-torch.tanh(
            (torch.sqrt(((self.X_c[:, 0] - x_c10) ** 2 + (self.X_c[:, 1] - y_c10) ** 2)) - r10) / (
                    math.sqrt(2) * 0.005)) + 1) * 0.5 + (-torch.tanh(
            (torch.sqrt(((self.X_c[:, 0] - x_c11) ** 2 + (self.X_c[:, 1] - y_c11) ** 2)) - r11) / (
                    math.sqrt(2) * 0.005)) + 1) * 0.5 + (-torch.tanh(
            (torch.sqrt(((self.X_c[:, 0] - x_c12) ** 2 + (self.X_c[:, 1] - y_c12) ** 2)) - r12) / (
                    math.sqrt(2) * 0.005)) + 1) * 0.5 + (-torch.tanh(
            (torch.sqrt(((self.X_c[:, 0] - x_c13) ** 2 + (self.X_c[:, 1] - y_c13) ** 2)) - r13) / (
                    math.sqrt(2) * 0.005)) + 1) * 0.5 + (-torch.tanh(
            (torch.sqrt(((self.X_c[:, 0] - x_c14) ** 2 + (self.X_c[:, 1] - y_c14) ** 2)) - r14) / (
                    math.sqrt(2) * 0.005)) + 1) * 0.5 + (-torch.tanh(
            (torch.sqrt(((self.X_c[:, 0] - x_c15) ** 2 + (self.X_c[:, 1] - y_c15) ** 2)) - r15) / (
                    math.sqrt(2) * 0.005)) + 1) * 0.5 + (-torch.tanh(
            (torch.sqrt(((self.X_c[:, 0] - x_c16) ** 2 + (self.X_c[:, 1] - y_c16) ** 2)) - r16) / (
                    math.sqrt(2) * 0.005)) + 1) * 0.5
        phi_c = torch.clamp(phi_c, 0, 1)
        loss_phi = self.criterion(phi, phi_c)

        loss = 10 * loss_phi

        loss.backward()

        if self.iter % 100 == 0:
            dcrease = -((loss.item() - self.loss_pre) / self.loss_pre) * 100
            self.loss_pre = loss.item()
            print(self.iter, loss.item(), f"dcrease{dcrease:.2f}%", 'CP Point:', self.X_c.size(0))
            print(f"loss_phi:{loss_phi.item()}")


        if self.iter % 10000 == 0:
            torch.save(net, 'Pear.pth')

        if self.iter % 1000 == 0:
            fig, ax = plt.subplots()
            ax.set_aspect('equal')
            X_c_current = self.X_c.cpu()
            X_c_current = X_c_current.detach().numpy()
            phi_np = phi.cpu().detach().numpy()
            pm = ax.scatter(X_c_current[:, 0], X_c_current[:, 1], c=phi_np, cmap='rainbow', marker='o', s=2, alpha=1,
                            edgecolors='none')
            cbar = fig.colorbar(pm, ax=ax)
            ax.set_title('Iter:{}'.format(self.iter), fontsize=15)
            plt.pause(0.5)
            plt.savefig('fig/Pear_{}.png'.format(self.iter))
            plt.close()

        writer.add_scalar('loss', loss.item(), self.iter)

        self.iter = self.iter + 1

        return loss

    def loss_func4(self):
        self.optimizer.zero_grad()

        Y_ub = self.model(self.X_ub)
        s12_ub = Y_ub[:, 3].reshape(-1, 1)
        s22_ub = Y_ub[:, 4].reshape(-1, 1)
        phi_ub = Y_ub[:, 5].reshape(-1, 1)
        loss_ub1 = self.criterion(s22_ub, self.t)
        loss_ub2 = self.criterion(s12_ub, self.zerobd)
        loss_phi1 = self.criterion(phi_ub, self.zerobd)

        Y_lb = self.model(self.X_lb)
        v_lb = Y_lb[:, 1].reshape(-1, 1)
        s12_lb = Y_lb[:, 3].reshape(-1, 1)
        phi_lb = Y_lb[:, 5].reshape(-1, 1)
        loss_lb1 = self.criterion(v_lb, self.zerobd)
        loss_lb2 = self.criterion(s12_lb, self.zerobd)

        loss_phi2 = self.criterion(phi_lb, self.zerobd)

        Y_lb2 = self.model(self.X_lb2)
        u_lb2 = Y_lb2[:, 0]
        v_lb2 = Y_lb2[:, 1]
        loss_lb3 = self.criterion(u_lb2, self.lb2) + self.criterion(v_lb2, self.lb2)

        Y_left = self.model(self.X_leftb)
        s11_left = Y_left[:, 2].reshape(-1, 1)
        s12_left = Y_left[:, 3].reshape(-1, 1)
        phi_left = Y_left[:, 5].reshape(-1, 1)
        loss_left = self.criterion(s11_left, self.zerobd) + self.criterion(s12_left, self.zerobd)
        loss_phi3 = self.criterion(phi_left, self.zerobd)

        Y_right = self.model(self.X_rightb)
        s11_right = Y_right[:, 2].reshape(-1, 1)
        s12_right = Y_right[:, 3].reshape(-1, 1)
        phi_right = Y_right[:, 5].reshape(-1, 1)
        loss_right = self.criterion(s11_right, self.zerobd) + self.criterion(s12_right, self.zerobd)
        loss_phi4 = self.criterion(phi_right, self.zerobd)

        Y_c = self.model(self.X_c)
        u = Y_c[:, 0].reshape(-1, 1)
        v = Y_c[:, 1].reshape(-1, 1)
        s11 = Y_c[:, 2]
        s12 = Y_c[:, 3]
        s22 = Y_c[:, 4]
        phi = Y_c[:, 5]
        # phi = torch.clamp(phi, 0, 1)
        x_c1 = self.lambda_1
        y_c1 = self.lambda_2
        r1 = torch.sqrt(self.lambda_3 ** 2)
        x_c2 = self.lambda_4
        y_c2 = self.lambda_5
        r2 = torch.sqrt(self.lambda_6 ** 2)
        x_c3 = self.lambda_7
        y_c3 = self.lambda_8
        r3 = torch.sqrt(self.lambda_9 ** 2)
        x_c4 = self.lambda_10
        y_c4 = self.lambda_11
        r4 = torch.sqrt(self.lambda_12 ** 2)
        x_c5 = self.lambda_13
        y_c5 = self.lambda_14
        r5 = torch.sqrt(self.lambda_15 ** 2)
        x_c6 = self.lambda_16
        y_c6 = self.lambda_17
        r6 = torch.sqrt(self.lambda_18 ** 2)
        x_c7 = self.lambda_19
        y_c7 = self.lambda_20
        r7 = torch.sqrt(self.lambda_21 ** 2)
        x_c8 = self.lambda_22
        y_c8 = self.lambda_23
        r8 = torch.sqrt(self.lambda_24 ** 2)
        x_c9 = self.lambda_25
        y_c9 = self.lambda_26
        r9 = torch.sqrt(self.lambda_27 ** 2)
        x_c10 = self.lambda_28
        y_c10 = self.lambda_29
        r10 = torch.sqrt(self.lambda_30 ** 2)
        x_c11 = self.lambda_31
        y_c11 = self.lambda_32
        r11 = torch.sqrt(self.lambda_33 ** 2)
        x_c12 = self.lambda_34
        y_c12 = self.lambda_35
        r12 = torch.sqrt(self.lambda_36 ** 2)
        x_c13 = self.lambda_37
        y_c13 = self.lambda_38
        r13 = torch.sqrt(self.lambda_39 ** 2)
        x_c14 = self.lambda_40
        y_c14 = self.lambda_41
        r14 = torch.sqrt(self.lambda_42 ** 2)
        x_c15 = self.lambda_43
        y_c15 = self.lambda_44
        r15 = torch.sqrt(self.lambda_45 ** 2)
        x_c16 = self.lambda_46
        y_c16 = self.lambda_47
        r16 = torch.sqrt(self.lambda_48 ** 2)
        phi_c = (-torch.tanh(
            (torch.sqrt(((self.X_c[:, 0] - x_c1) ** 2 + (self.X_c[:, 1] - y_c1) ** 2)) - r1) / (
                    math.sqrt(2) * 0.005)) + 1) * 0.5 + (-torch.tanh(
            (torch.sqrt(((self.X_c[:, 0] - x_c2) ** 2 + (self.X_c[:, 1] - y_c2) ** 2)) - r2) / (
                    math.sqrt(2) * 0.005)) + 1) * 0.5 + (-torch.tanh(
            (torch.sqrt(((self.X_c[:, 0] - x_c3) ** 2 + (self.X_c[:, 1] - y_c3) ** 2)) - r3) / (
                    math.sqrt(2) * 0.005)) + 1) * 0.5 + (-torch.tanh(
            (torch.sqrt(((self.X_c[:, 0] - x_c4) ** 2 + (self.X_c[:, 1] - y_c4) ** 2)) - r4) / (
                    math.sqrt(2) * 0.005)) + 1) * 0.5 + (-torch.tanh(
            (torch.sqrt(((self.X_c[:, 0] - x_c5) ** 2 + (self.X_c[:, 1] - y_c5) ** 2)) - r5) / (
                    math.sqrt(2) * 0.005)) + 1) * 0.5 + (-torch.tanh(
            (torch.sqrt(((self.X_c[:, 0] - x_c6) ** 2 + (self.X_c[:, 1] - y_c6) ** 2)) - r6) / (
                    math.sqrt(2) * 0.005)) + 1) * 0.5 + (-torch.tanh(
            (torch.sqrt(((self.X_c[:, 0] - x_c7) ** 2 + (self.X_c[:, 1] - y_c7) ** 2)) - r7) / (
                    math.sqrt(2) * 0.005)) + 1) * 0.5 + (-torch.tanh(
            (torch.sqrt(((self.X_c[:, 0] - x_c8) ** 2 + (self.X_c[:, 1] - y_c8) ** 2)) - r8) / (
                    math.sqrt(2) * 0.005)) + 1) * 0.5 + (-torch.tanh(
            (torch.sqrt(((self.X_c[:, 0] - x_c9) ** 2 + (self.X_c[:, 1] - y_c9) ** 2)) - r9) / (
                    math.sqrt(2) * 0.005)) + 1) * 0.5 + (-torch.tanh(
            (torch.sqrt(((self.X_c[:, 0] - x_c10) ** 2 + (self.X_c[:, 1] - y_c10) ** 2)) - r10) / (
                    math.sqrt(2) * 0.005)) + 1) * 0.5 + (-torch.tanh(
            (torch.sqrt(((self.X_c[:, 0] - x_c11) ** 2 + (self.X_c[:, 1] - y_c11) ** 2)) - r11) / (
                    math.sqrt(2) * 0.005)) + 1) * 0.5 + (-torch.tanh(
            (torch.sqrt(((self.X_c[:, 0] - x_c12) ** 2 + (self.X_c[:, 1] - y_c12) ** 2)) - r12) / (
                    math.sqrt(2) * 0.005)) + 1) * 0.5 + (-torch.tanh(
            (torch.sqrt(((self.X_c[:, 0] - x_c13) ** 2 + (self.X_c[:, 1] - y_c13) ** 2)) - r13) / (
                    math.sqrt(2) * 0.005)) + 1) * 0.5 + (-torch.tanh(
            (torch.sqrt(((self.X_c[:, 0] - x_c14) ** 2 + (self.X_c[:, 1] - y_c14) ** 2)) - r14) / (
                    math.sqrt(2) * 0.005)) + 1) * 0.5 + (-torch.tanh(
            (torch.sqrt(((self.X_c[:, 0] - x_c15) ** 2 + (self.X_c[:, 1] - y_c15) ** 2)) - r15) / (
                    math.sqrt(2) * 0.005)) + 1) * 0.5 + (-torch.tanh(
            (torch.sqrt(((self.X_c[:, 0] - x_c16) ** 2 + (self.X_c[:, 1] - y_c16) ** 2)) - r16) / (
                    math.sqrt(2) * 0.005)) + 1) * 0.5
        phi_c = torch.clamp(phi_c, 0, 1)
        loss_phi = self.criterion(phi, phi_c)

        du_dX = torch.autograd.grad(inputs=self.X_c, outputs=u, grad_outputs=torch.ones_like(u), retain_graph=True,
                                    create_graph=True)[0]
        dv_dX = torch.autograd.grad(inputs=self.X_c, outputs=v, grad_outputs=torch.ones_like(v), retain_graph=True,
                                    create_graph=True)[0]
        du_dx = du_dX[:, 0]
        du_dy = du_dX[:, 1]
        dv_dx = dv_dX[:, 0]
        dv_dy = dv_dX[:, 1]

        e11 = du_dx
        e22 = dv_dy
        e12 = 0.5 * (du_dy + dv_dx)

        s11_p = (E / (1 - miu ** 2)) * (e11 + miu * e22)
        s22_p = (E / (1 - miu ** 2)) * (e22 + miu * e11)
        s12_p = (E / (1 + miu)) * e12

        loss_f1 = self.criterion(s11, s11_p)
        loss_f2 = self.criterion(s12, s12_p)
        loss_f3 = self.criterion(s22, s22_p)

        ds11_dx = \
            torch.autograd.grad(inputs=self.X_c, outputs=s11, grad_outputs=torch.ones_like(s11), retain_graph=True,
                                create_graph=True)[0][:, 0]
        ds12_dx = \
            torch.autograd.grad(inputs=self.X_c, outputs=s12, grad_outputs=torch.ones_like(s12), retain_graph=True,
                                create_graph=True)[0][:, 0]
        ds12_dy = \
            torch.autograd.grad(inputs=self.X_c, outputs=s12, grad_outputs=torch.ones_like(s12), retain_graph=True,
                                create_graph=True)[0][:, 1]
        ds22_dy = \
            torch.autograd.grad(inputs=self.X_c, outputs=s22, grad_outputs=torch.ones_like(s22), retain_graph=True,
                                create_graph=True)[0][:, 1]
        g_phi = (1 - phi) ** 2

        dphi_dx = \
            torch.autograd.grad(inputs=self.X_c, outputs=phi, grad_outputs=torch.ones_like(phi), retain_graph=True,
                                create_graph=True)[0][:, 0]
        dphi_dy = \
            torch.autograd.grad(inputs=self.X_c, outputs=phi, grad_outputs=torch.ones_like(phi), retain_graph=True,
                                create_graph=True)[0][:, 1]
        dg_dx = 2 * (phi - 1) * dphi_dx
        dg_dy = 2 * (phi - 1) * dphi_dy
        loss_pde1 = self.criterion(dg_dx * s11 + dg_dy * s12, -ds11_dx * g_phi - ds12_dy * g_phi)
        loss_pde2 = self.criterion(dg_dx * s12 + dg_dy * s22, -ds12_dx * g_phi - ds22_dy * g_phi)
        loss_pde = loss_pde1 + loss_pde2
        loss_bc = 1 * loss_ub1 + (1 * loss_lb1 + loss_lb2 + loss_lb3 + loss_ub2 + loss_left + loss_right)
        loss_f = loss_f1 + loss_f2 + loss_f3
        loss_phi_bc = loss_phi1 + loss_phi2 + loss_phi3 + loss_phi4

        loss = 1 * loss_ub1 + 10 * (
                1 * loss_lb1 + loss_lb2 + loss_lb3 + loss_ub2 + loss_left + loss_right) + 1 * loss_f + loss_pde + loss_phi_bc + 10 * loss_phi
        loss.backward()

        if self.iter % 100 == 0:
            dcrease = -((loss.item() - self.loss_pre) / self.loss_pre) * 100
            self.loss_pre = loss.item()
            print(self.iter, loss.item(), f"dcrease{dcrease:.2f}%", 'CP Point:', self.X_c.size(0))
            print(f"loss_BC1:{loss_ub1.item()}")
            print(
                f"loss_BC2:{1 * loss_lb1 + loss_left + loss_right}")
            print(f"loss_f:{loss_f.item()}")
            print(f"loss_pde:{loss_pde1.item() + loss_pde2.item()}")
            print(f"loss_phi_bc:{loss_phi1.item() + loss_phi2.item() + loss_phi3.item() + loss_phi4.item()}")
            print(f"loss_phi:{loss_phi.item()}")


        if self.iter % 1000 == 0:
            fig, ax = plt.subplots()
            ax.set_aspect('equal')
            X_c_current = self.X_c.cpu()
            X_c_current = X_c_current.detach().numpy()
            phi_np = phi.cpu().detach().numpy()
            pm = ax.scatter(X_c_current[:, 0], X_c_current[:, 1], c=phi_np, cmap='rainbow', marker='o', s=2, alpha=1,
                            edgecolors='none')
            cbar = fig.colorbar(pm, ax=ax)
            ax.set_title('Iter:{}'.format(self.iter), fontsize=15)
            plt.pause(0.5)
            plt.savefig('fig/Pear_{}.png'.format(self.iter))
            plt.close()

        if self.iter % 10000 == 0:
            torch.save(net, 'Pear.pth')

        writer.add_scalar('loss', loss.item(), self.iter)
        writer.add_scalar('loss_pde', loss_pde.item(), self.iter)
        writer.add_scalar('loss_f', loss_f.item(), self.iter)
        writer.add_scalar('loss_bc', loss_bc.item(), self.iter)
        writer.add_scalar('loss_phi', loss_phi.item(), self.iter)


        self.iter = self.iter + 1

        return loss

    def train1(self):
        for i in range(2000):
            self.adam.step(self.loss_func)
        self.optimizer.step(self.loss_func)

    def train2(self):
        for i in range(2000):
            self.adam.step(self.loss_func2)
        self.optimizer.step(self.loss_func2)

    def train3(self):
        for i in range(2000):
            self.adam.step(self.loss_func3)
        self.optimizer2.step(self.loss_func3)

    def train4(self):
        for i in range(2000):
            self.adam.step(self.loss_func4)
        self.optimizer3.step(self.loss_func4)


net = PINN()
# net.model.load_state_dict(torch.load('Pear_pretrain_param.pth'))

path = "fig"
if os.path.exists(path):
    shutil.rmtree(path)
else:
    pass
os.makedirs(path)

plt.ion()
net.train1()
torch.save(net, 'Step1.pth')
torch.save(net.model.state_dict(), 'Pear_pretrain_param.pth')
for para in net.model.net1.parameters():
    para.requires_grad = False
for para in net.model.net2.parameters():
    para.requires_grad = False
for para in net.model.net3.parameters():
    para.requires_grad = False
net.train3()
torch.save(net, 'Step2.pth')

for para in net.model.net3.parameters():
    para.requires_grad = True
net.lambda_1.requires_grad = False
net.lambda_2.requires_grad = False
net.lambda_3.requires_grad = False

for para in net.model.net1.parameters():
    para.requires_grad = True
for para in net.model.net2.parameters():
    para.requires_grad = True
net.train4()
torch.save(net, 'Step3.pth')
net.lambda_1.requires_grad = True
net.lambda_2.requires_grad = True
net.lambda_3.requires_grad = True
net.train2()
torch.save(net, 'Step4.pth')
plt.ioff()
print('done')
exit()