# Hyperelastic
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
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
# device = torch.device('cuda')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"use {device} to compute")
writer = SummaryWriter('./path/to/log/Hyperelastic')

lamb = 15
miu = 5
k=0.0000001

class DNN1(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, hidden_num):
        super(DNN1, self).__init__()
        # self.B =B
        self.net = nn.Sequential()
        self.net.add_module('Linear_layer_1', nn.Linear(input_size, hidden_size))
        self.net.add_module('Tanh_layer_1', nn.Tanh())
        for num in range(2, hidden_num+1):
            self.net.add_module('Linear_layer_%d' % (num), nn.Linear(hidden_size, hidden_size))  # Linear layer
            self.net.add_module('Tanh_layer_%d' % (num), nn.Tanh())  # Activation Layer
        self.net.add_module('Linear_layer_final', nn.Linear(hidden_size, output_size))  # Output Layer

    # Forward Feed
    def forward(self, x):
        return self.net(x)


class DNN2(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, hidden_num):
        super(DNN2, self).__init__()
        # self.B = B
        self.net = nn.Sequential()
        self.net.add_module('Linear_layer_1', nn.Linear(input_size, hidden_size))
        self.net.add_module('Tanh_layer_1', nn.Tanh())
        for num in range(2, hidden_num+1):
            self.net.add_module('Linear_layer_%d' % (num), nn.Linear(hidden_size, hidden_size))  # Linear layer
            self.net.add_module('Tanh_layer_%d' % (num), nn.Tanh())  # Activation Layer
        self.net.add_module('Linear_layer_final', nn.Linear(hidden_size, output_size))  # Output Layer

    # Forward Feed
    def forward(self, x):
        # xb = map_x(x, self.B)
        return self.net(x)


class DNN3(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, hidden_num):
        super(DNN3, self).__init__()
        # self.B = B
        self.net = nn.Sequential()
        self.net.add_module('Linear_layer_1', nn.Linear(input_size, hidden_size))
        self.net.add_module('Tanh_layer_1', nn.Tanh())
        for num in range(2, hidden_num+1):
            self.net.add_module('Linear_layer_%d' % (num), nn.Linear(hidden_size, hidden_size))  # Linear layer
            self.net.add_module('Tanh_layer_%d' % (num), nn.Tanh())  # Activation Layer
        self.net.add_module('Linear_layer_final', nn.Linear(hidden_size, output_size))  # Output Layer

    # Forward Feed
    def forward(self, x):
        return self.net(x)


class DNN4(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, hidden_num):
        super(DNN4, self).__init__()
        # self.B = B
        self.net = nn.Sequential()
        self.net.add_module('Linear_layer_1', nn.Linear(input_size, hidden_size))
        self.net.add_module('Tanh_layer_1', nn.Tanh())
        for num in range(2, hidden_num+1):
            self.net.add_module('Linear_layer_%d' % (num), nn.Linear(hidden_size, hidden_size))  # Linear layer
            self.net.add_module('Tanh_layer_%d' % (num), nn.Tanh())  # Activation Layer
        self.net.add_module('Linear_layer_final', nn.Linear(hidden_size, output_size))  # Output Layer

    # Forward Feed
    def forward(self, x):
        return self.net(x)



class ConcatNet(nn.Module):
    def __init__(self, net1, net2, net3, net4):
        super(ConcatNet, self).__init__()
        self.net1 = net1
        self.net2 = net2
        self.net3 = net3
        self.net4 = net4

    def forward(self, x):
        out1 = self.net1(x)
        out2 = self.net2(x)
        out3 = self.net3(x)
        out4 = self.net4(x)
        out = torch.cat((out1, out2), dim=1)
        out = torch.cat((out, out3), dim=1)
        out = torch.cat((out, out4), dim=1)
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


def gradients(inputs, outputs):
    return torch.autograd.grad(inputs=inputs, outputs=outputs, grad_outputs=torch.ones_like(outputs), retain_graph=True,
                               create_graph=True)


def map_x(x,B):
    xp = torch.matmul(2*math.pi*x,B)
    return torch.cat([torch.sin(xp),torch.cos(xp)],dim=-1)


def GenHoleSurfPT(xc, yc, a, b, N_PT):
    # Generate
    theta = lhs(1, N_PT)
    theta = theta * np.pi
    xx = np.multiply(a, np.cos(theta)) + xc
    yy = np.multiply(b, np.sin(theta)) + yc
    xx = xx.flatten()[:, None]
    yy = yy.flatten()[:, None]
    return xx, yy


def DelHolePT(XYT_c, xc=0, yc=0, a=0, b=0, r=0.1):
    # Delete points within hole
    dst = np.array([(((xyt[0] - xc)/a) ** 2 + ((xyt[1] - yc)/b) ** 2) for xyt in XYT_c])
    return XYT_c[dst > r, :]


class PINN:
    def __init__(self):
        # B = torch.randn(2, 256).cuda() * 0.1
        net1 = DNN1(input_size=2, hidden_size=60, output_size=2, hidden_num=6).to(device)
        net2 = DNN2(input_size=2, hidden_size=120, output_size=4, hidden_num=6).to(device)
        net3 = DNN3(input_size=2, hidden_size=40, output_size=1, hidden_num=6).to(device)
        net4 = DNN4(input_size=2, hidden_size=40, output_size=1, hidden_num=6).to(device)
        self.model = ConcatNet(net1, net2, net3, net4)
        ### Use DataParallel ###
        # if torch.cuda.device_count() > 1:
            # print("Let's use", torch.cuda.device_count(), "GPUs!")
        # self.model = nn.DataParallel(self.model)
        self.model.apply(weight_init)
        print(self.model)
        # data gernerate
        bounds1 = [[-0.7, -0.4], [0.7, 0.4]]
        init_data1 = HammersleySampling(data_input=bounds1, number_of_samples=40000)
        data_c = init_data1.sample_points()
        data_c = DelHolePT(XYT_c=data_c, xc=0.0, yc=-0.4, a=0.3, b=0.4, r=1)
        x_c = data_c[:, 0]
        y_c = data_c[:, 1]
        X_c = np.hstack((x_c.reshape(-1, 1), y_c.reshape(-1, 1)))

        edge1_x = -0.7 + 1.4 * lhs(1, 280)
        edge1_y = 0.4 * np.ones_like(edge1_x)
        edge2_y = -0.4 + 0.8 * lhs(1, 160)
        edge2_x = -0.7 * np.ones_like(edge2_y)
        edge3_y = -0.4 + 0.8 * lhs(1, 160)
        edge3_x = 0.7 * np.ones_like(edge3_y)
        edge4_x = -0.7 + 0.4 * lhs(1, 80)
        edge4_y = -0.4 * np.ones_like(edge4_x)
        edge5_x = 0.3 + 0.4 * lhs(1, 80)
        edge5_y = -0.4 * np.ones_like(edge5_x)
        edge6_x, edge6_y = GenHoleSurfPT(xc=0.0, yc=-0.4, a=0.3, b=0.4, N_PT=300)
        X_e1 = np.hstack((edge1_x, edge1_y))
        X_e2 = np.hstack((edge2_x, edge2_y))
        X_e3 = np.hstack((edge3_x, edge3_y))
        X_e4 = np.hstack((edge4_x, edge4_y))
        X_e5 = np.hstack((edge5_x, edge5_y))
        X_e6 = np.hstack((edge6_x, edge6_y))

        s1 = np.zeros((280, 1))
        s2 = np.zeros((160, 1))
        s3 = np.zeros((160, 1))
        s4 = np.zeros((80, 1))
        s5 = np.zeros((80, 1))
        s6 = np.zeros((300, 1))
        t = 2 * np.ones((280, 1))

        X_lb2 = torch.tensor((-0.7, -0.4)).reshape(1, 2)
        self.X_lb2 = X_lb2
        self.X_lb2 = self.X_lb2.to(device)
        lb2 = torch.zeros((1,))
        self.lb2 = lb2
        self.lb2 = self.lb2.to(device)

        X_c = torch.tensor(X_c)
        self.X_c = X_c
        self.X_c = self.X_c.to(device)
        X_e1 = torch.tensor(X_e1)
        self.X_e1 = X_e1
        self.X_e1 = self.X_e1.to(device)
        X_e2 = torch.tensor(X_e2)
        self.X_e2 = X_e2
        self.X_e2 = self.X_e2.to(device)
        X_e3 = torch.tensor(X_e3)
        self.X_e3 = X_e3
        self.X_e3 = self.X_e3.to(device)
        X_e4 = torch.tensor(X_e4)
        self.X_e4 = X_e4
        self.X_e4 = self.X_e4.to(device)
        X_e5 = torch.tensor(X_e5)
        self.X_e5 = X_e5
        self.X_e5 = self.X_e5.to(device)
        X_e6 = torch.tensor(X_e6)
        self.X_e6 = X_e6
        self.X_e6 = self.X_e6.to(device)
        s1 = torch.tensor(s1)
        self.s1 = s1
        self.s1 = self.s1.to(device)
        s2 = torch.tensor(s2)
        self.s2 = s2
        self.s2 = self.s2.to(device)
        s3 = torch.tensor(s3)
        self.s3 = s3
        self.s3 = self.s3.to(device)
        s4 = torch.tensor(s4)
        self.s4 = s4
        self.s4 = self.s4.to(device)
        s5 = torch.tensor(s5)
        self.s5 = s5
        self.s5 = self.s5.to(device)
        s6 = torch.tensor(s6)
        self.s6 = s6
        self.s6 = self.s6.to(device)
        t = torch.tensor(t)
        self.t = t
        self.t = self.t.to(device)

        path = 'Hyperelastic.csv'
        in_data = pd.read_csv(path)
        in_data = in_data.values
        x_data = in_data[:, 0].reshape(-1, 1)
        y_data = in_data[:, 1].reshape(-1, 1)
        u_data = in_data[:, 2].reshape(-1, 1)
        v_data = in_data[:, 3].reshape(-1, 1)
        e11_data = in_data[:, 4].reshape(-1, 1)
        e12_data = in_data[:, 5].reshape(-1, 1)
        e22_data = in_data[:, 6].reshape(-1, 1)

        X_data = np.hstack((x_data, y_data))
        X_data = torch.tensor(X_data)
        self.X_data = X_data
        self.X_data = self.X_data.to(device)
        u_data = torch.tensor(u_data)
        self.u_data = u_data
        self.u_data = self.u_data.to(device)
        v_data = torch.tensor(v_data)
        self.v_data = v_data
        self.v_data = self.v_data.to(device)
        e11_data = torch.tensor(e11_data)
        self.e11_data = e11_data
        self.e11_data = self.e11_data.to(device)
        e12_data = torch.tensor(e12_data)
        self.e12_data = e12_data
        self.e12_data = self.e12_data.to(device)
        e22_data = torch.tensor(e22_data)
        self.e22_data = e22_data
        self.e22_data = self.e22_data.to(device)

        a1 = np.random.rand() * 0.16 - 0.24
        b1 = np.random.rand() * 0.16 + 0.1
        self.lambda_1 = torch.tensor([0.43], requires_grad=True).to(device)
        self.lambda_2 = torch.tensor([0.05], requires_grad=True).to(device)
        self.lambda_3 = torch.tensor([0.04], requires_grad=True).to(device)
        self.lambda_1 = torch.nn.Parameter(self.lambda_1)
        self.lambda_2 = torch.nn.Parameter(self.lambda_2)
        self.lambda_3 = torch.nn.Parameter(self.lambda_3)
        self.model.register_parameter('lambda_1', self.lambda_1)
        self.model.register_parameter('lambda_2', self.lambda_2)
        self.model.register_parameter('lambda_3', self.lambda_3)

        self.X_c.requires_grad = True
        self.X_data.requires_grad = True

        self.loss_pre = 10
        # optimizer
        self.criterion = torch.nn.MSELoss()
        # self.criterion = torch.nn.L1Loss()
        self.iter = 0
        self.optimizer = torch.optim.LBFGS(
            self.model.parameters(),
            lr=1,
            max_iter=198000,
            max_eval=198000,
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
        self.adam = torch.optim.Adam(self.model.parameters(), lr=0.00001)


# def loss
    def loss_func(self):
        self.optimizer.zero_grad()
        # B = torch.randn(2, 256).cuda() * 10
        Y_1 = self.model(self.X_e1)
        P12_1 = Y_1[:, 3].reshape(-1, 1)
        P22_1 = Y_1[:, 5].reshape(-1, 1)
        phi_1 = Y_1[:, 7].reshape(-1, 1)
        loss_t = self.criterion(P22_1, self.t)
        loss_bd1 = self.criterion(P12_1, self.s1)
        loss_phi1 = self.criterion(phi_1, self.s1)

        Y_2 = self.model(self.X_e2)
        P11_2 = Y_2[:, 2].reshape(-1, 1)
        P21_2 = Y_2[:, 4].reshape(-1, 1)
        phi_2 = Y_2[:, 7].reshape(-1, 1)
        loss_bd2 = self.criterion(P11_2, self.s2) + self.criterion(P21_2, self.s2)
        loss_phi2 = self.criterion(phi_2, self.s2)

        Y_3 = self.model(self.X_e3)
        P11_3 = Y_3[:, 2].reshape(-1, 1)
        P21_3 = Y_3[:, 4].reshape(-1, 1)
        phi_3 = Y_3[:, 7].reshape(-1, 1)
        loss_bd3 = self.criterion(P11_3, self.s3) + self.criterion(P21_3, self.s3)
        loss_phi3 = self.criterion(phi_3, self.s3)

        Y_4 = self.model(self.X_e4)
        v_4 = Y_4[:, 1].reshape(-1, 1)
        P12_4 = Y_4[:, 3].reshape(-1, 1)
        P21_4 = Y_4[:, 4].reshape(-1, 1)
        phi_4 = Y_4[:, 7].reshape(-1, 1)
        loss_bd4 = self.criterion(v_4, self.s4) + self.criterion(P12_4, self.s4)
        loss_phi4 = self.criterion(phi_4, self.s4)

        Y_5 = self.model(self.X_e5)
        v_5 = Y_5[:, 1].reshape(-1, 1)
        P12_5 = Y_5[:, 3].reshape(-1, 1)
        P21_5 = Y_5[:, 4].reshape(-1, 1)
        phi_5 = Y_5[:, 7].reshape(-1, 1)
        loss_bd5 = self.criterion(v_5, self.s5) + self.criterion(P12_5, self.s5)
        loss_phi5 = self.criterion(phi_5, self.s5)

        Y_6 = self.model(self.X_e6)
        P11_6 = Y_6[:, 2].reshape(-1, 1)
        P12_6 = Y_6[:, 3].reshape(-1, 1)
        P21_6 = Y_6[:, 4].reshape(-1, 1)
        P22_6 = Y_6[:, 5].reshape(-1, 1)
        phi_6 = Y_6[:, 7].reshape(-1, 1)
        x_6 = self.X_e6[:, 0].reshape(-1, 1)
        y_6 = self.X_e6[:, 1].reshape(-1, 1)
        loss_phi6 = self.criterion(phi_6, self.s6)
        loss_bd6 = self.criterion(P11_6 * x_6 * (2 / 0.09), -P12_6 * (y_6 + 0.4) * (2 / 0.16)) + self.criterion(
            P21_6 * x_6 * (2 / 0.09), -P22_6 * (y_6 + 0.4) * (2 / 0.16))

        Y_lb2 = self.model(self.X_lb2)
        u_lb2 = Y_lb2[:, 0]
        v_lb2 = Y_lb2[:, 1]
        loss_lb2 = self.criterion(u_lb2, self.lb2) + self.criterion(v_lb2, self.lb2)

        Y_c = self.model(self.X_c)
        u = Y_c[:, 0].reshape(-1, 1)
        v = Y_c[:, 1].reshape(-1, 1)
        P11 = Y_c[:, 2]
        P12 = Y_c[:, 3]
        P21 = Y_c[:, 4]
        P22 = Y_c[:, 5]
        p = Y_c[:, 6]
        phi = Y_c[:, 7]

        du_dX = torch.autograd.grad(inputs=self.X_c, outputs=u, grad_outputs=torch.ones_like(u), retain_graph=True,
                                    create_graph=True)[0]
        dv_dX = torch.autograd.grad(inputs=self.X_c, outputs=v, grad_outputs=torch.ones_like(v), retain_graph=True,
                                    create_graph=True)[0]
        gradU = torch.cat((du_dX, dv_dX), dim=1)
        size = gradU.size(0)
        gradU = gradU.reshape(size, 2, 2)
        I = torch.zeros_like(gradU).to(device)
        I[:, 0, 0] = 1
        I[:, 1, 1] = 1
        g_phi = (1 - phi) ** 2
        g_phi2 = torch.unsqueeze(g_phi.reshape(-1, 1), dim=-1)
        g_phi2 = torch.cat((g_phi2, g_phi2), dim=1)
        g_phi2 = torch.cat((g_phi2, g_phi2), dim=2)
        gradU = gradU * g_phi2
        F = gradU + I
        J = torch.det(F)
        J_star = torch.ones_like(J).to(device)
        loss_J = self.criterion(J, J_star)
        FT = torch.transpose(F, dim0=1, dim1=2)
        FTi = torch.inverse(FT)
        p = p.reshape(-1, 1)
        p = torch.unsqueeze(p, dim=-1)
        p = torch.cat((p, p), dim=1)
        p = torch.cat((p, p), dim=2)
        P = miu * F - p * FTi

        P11_star = P[:, 0, 0]
        P12_star = P[:, 0, 1]
        P21_star = P[:, 1, 0]
        P22_star = P[:, 1, 1]

        P11 = P11 * g_phi
        P12 = P12 * g_phi
        P21 = P21 * g_phi
        P22 = P22 * g_phi


        loss_P = self.criterion(P11, P11_star) + self.criterion(P12, P12_star) + self.criterion(P21, P21_star) + self.criterion(P22, P22_star)

        dP11_dx = \
            torch.autograd.grad(inputs=self.X_c, outputs=P11, grad_outputs=torch.ones_like(P11), retain_graph=True,
                                create_graph=True)[0][:, 0]
        dP12_dy = \
            torch.autograd.grad(inputs=self.X_c, outputs=P12, grad_outputs=torch.ones_like(P12), retain_graph=True,
                                create_graph=True)[0][:, 1]
        dP21_dx = \
            torch.autograd.grad(inputs=self.X_c, outputs=P21, grad_outputs=torch.ones_like(P21), retain_graph=True,
                                create_graph=True)[0][:, 0]
        dP22_dy = \
            torch.autograd.grad(inputs=self.X_c, outputs=P22, grad_outputs=torch.ones_like(P22), retain_graph=True,
                                create_graph=True)[0][:, 1]


        loss_pde1 = self.criterion(dP11_dx, - dP12_dy)
        loss_pde2 = self.criterion(dP21_dx, - dP22_dy)

        Y_in = self.model(self.X_data)
        u_in = Y_in[:, 0]
        v_in = Y_in[:, 1]
        phi_in = Y_in[:, 7]
        du_in_dX = \
            torch.autograd.grad(inputs=self.X_data, outputs=u_in, grad_outputs=torch.ones_like(u_in), retain_graph=True,
                                create_graph=True)[0]
        dv_in_dX = \
            torch.autograd.grad(inputs=self.X_data, outputs=v_in, grad_outputs=torch.ones_like(v_in), retain_graph=True,
                                create_graph=True)[0]
        gradU_in = torch.cat((du_in_dX, dv_in_dX), dim=1)
        size_in = gradU_in.size(0)
        gradU_in = gradU_in.reshape(size_in, 2, 2)
        I_in = torch.zeros_like(gradU_in).to(device)
        I_in[:, 0, 0] = 1
        I_in[:, 1, 1] = 1
        g_phi_in = (1 - phi_in) ** 2
        g_phi2_in = torch.unsqueeze(g_phi_in.reshape(-1, 1), dim=-1)
        g_phi2_in = torch.cat((g_phi2_in, g_phi2_in), dim=1)
        g_phi2_in = torch.cat((g_phi2_in, g_phi2_in), dim=2)
        gradU_in = gradU_in * g_phi2_in
        F_in = gradU_in + I_in
        # F_in = F[self.index,:]
        F11_in = F_in[:, 0, 0]
        F12_in = F_in[:, 0, 1]
        F21_in = F_in[:, 1, 0]
        F22_in = F_in[:, 1, 1]
        C11_in = F11_in * F11_in + F21_in * F21_in
        C12_in = F11_in * F12_in + F21_in * F22_in
        C22_in = F12_in * F12_in + F22_in * F22_in
        e11_in = 0.5 * (-1 + C11_in)
        e12_in = 0.5 * C12_in
        e22_in = 0.5 * (-1 + C22_in)
        loss_data = self.criterion(e11_in.reshape(-1, 1), self.e11_data) + 1 * self.criterion(e12_in.reshape(-1, 1),
                                                                                               self.e12_data) + self.criterion(
            e22_in.reshape(-1, 1), self.e22_data)

        loss_pde = loss_pde1 + loss_pde2
        loss_bc = loss_bd1 + loss_bd2 + loss_bd3 + loss_bd4 + loss_bd5 + loss_bd6 + loss_lb2
        loss_phi_bc = loss_phi1 + loss_phi2 + loss_phi3 + loss_phi4 + loss_phi5 + loss_phi6
        loss = 10 * loss_bc + loss_t + 3 * loss_P + loss_pde + 50 * loss_phi_bc + 50 * loss_data + 10 * loss_J
        loss.backward()


        if self.iter % 100 == 0:
            dcrease = -((loss.item() - self.loss_pre) / self.loss_pre) * 100
            self.loss_pre = loss.item()
            print(self.iter, loss.item(), f"dcrease{dcrease:.2f}%", 'CP Point:', self.X_c.size(0))
            print(f"loss_BC:{loss_bc.item()}")
            print(f"loss_t:{loss_t.item()}")
            print(f"loss_P:{loss_P.item()}")
            print(f"loss_pde:{loss_pde1.item() + loss_pde2.item()}")
            print(f"loss_phi_bc:{loss_phi_bc.item()}")
            print(f"loss_J:{loss_J.item()}")
            print(f"loss_data:{loss_data.item()}")


        if self.iter % 1000 == 0:
            fig, ax = plt.subplots()
            ax.set_aspect('equal')
            X_c_current = self.X_c.cpu()
            X_c_current = X_c_current.detach().numpy()
            J_np = phi.cpu().detach().numpy()
            pm = ax.scatter(X_c_current[:, 0], X_c_current[:, 1], c=J_np, cmap='rainbow', marker='o', s=2, alpha=1, edgecolors='none')
            cbar = fig.colorbar(pm, ax=ax)
            ax.set_title('Iter:{}'.format(self.iter), fontsize=15)
            plt.pause(0.5)
            plt.savefig('fig/Hyperelastic_{}.png'.format(self.iter))
            plt.close()


        if self.iter % 10000 == 0:
            torch.save(net, 'Hyperelastic.pth')

        writer.add_scalar('loss', loss.item(), self.iter)
        writer.add_scalar('loss_pde', loss_pde.item(), self.iter)
        writer.add_scalar('loss_P', loss_P.item(), self.iter)
        writer.add_scalar('loss_J', loss_J.item(), self.iter)
        writer.add_scalar('loss_bc', loss_bc.item(), self.iter)



        self.iter = self.iter + 1

        return loss


    def loss_func2(self):
        self.optimizer.zero_grad()
        Y_1 = self.model(self.X_e1)
        P12_1 = Y_1[:, 3].reshape(-1, 1)
        P22_1 = Y_1[:, 5].reshape(-1, 1)
        phi_1 = Y_1[:, 7].reshape(-1, 1)
        loss_t = self.criterion(P22_1, self.t)
        loss_bd1 = self.criterion(P12_1, self.s1)
        loss_phi1 = self.criterion(phi_1, self.s1)

        Y_2 = self.model(self.X_e2)
        P11_2 = Y_2[:, 2].reshape(-1, 1)
        P21_2 = Y_2[:, 4].reshape(-1, 1)
        phi_2 = Y_2[:, 7].reshape(-1, 1)
        loss_bd2 = self.criterion(P11_2, self.s2) + self.criterion(P21_2, self.s2)
        loss_phi2 = self.criterion(phi_2, self.s2)

        Y_3 = self.model(self.X_e3)
        P11_3 = Y_3[:, 2].reshape(-1, 1)
        P21_3 = Y_3[:, 4].reshape(-1, 1)
        phi_3 = Y_3[:, 7].reshape(-1, 1)
        loss_bd3 = self.criterion(P11_3, self.s3) + self.criterion(P21_3, self.s3)
        loss_phi3 = self.criterion(phi_3, self.s3)

        Y_4 = self.model(self.X_e4)
        v_4 = Y_4[:, 1].reshape(-1, 1)
        P12_4 = Y_4[:, 3].reshape(-1, 1)
        P21_4 = Y_4[:, 4].reshape(-1, 1)
        phi_4 = Y_4[:, 7].reshape(-1, 1)
        loss_bd4 = self.criterion(v_4, self.s4) + self.criterion(P12_4, self.s4)
        loss_phi4 = self.criterion(phi_4, self.s4)

        Y_5 = self.model(self.X_e5)
        v_5 = Y_5[:, 1].reshape(-1, 1)
        P12_5 = Y_5[:, 3].reshape(-1, 1)
        P21_5 = Y_5[:, 4].reshape(-1, 1)
        phi_5 = Y_5[:, 7].reshape(-1, 1)
        loss_bd5 = self.criterion(v_5, self.s5) + self.criterion(P12_5, self.s5)
        loss_phi5 = self.criterion(phi_5, self.s5)

        Y_6 = self.model(self.X_e6)
        P11_6 = Y_6[:, 2].reshape(-1, 1)
        P12_6 = Y_6[:, 3].reshape(-1, 1)
        P21_6 = Y_6[:, 4].reshape(-1, 1)
        P22_6 = Y_6[:, 5].reshape(-1, 1)
        phi_6 = Y_6[:, 7].reshape(-1, 1)
        x_6 = self.X_e6[:, 0].reshape(-1, 1)
        y_6 = self.X_e6[:, 1].reshape(-1, 1)
        loss_phi6 = self.criterion(phi_6, self.s6)
        loss_bd6 = self.criterion(P11_6 * x_6 * (2 / 0.09), -P12_6 * (y_6 + 0.4) * (2 / 0.16)) + self.criterion(
            P21_6 * x_6 * (2 / 0.09), -P22_6 * (y_6 + 0.4) * (2 / 0.16))

        Y_lb2 = self.model(self.X_lb2)
        u_lb2 = Y_lb2[:, 0]
        v_lb2 = Y_lb2[:, 1]
        loss_lb2 = self.criterion(u_lb2, self.lb2) + self.criterion(v_lb2, self.lb2)

        Y_c = self.model(self.X_c)
        u = Y_c[:, 0].reshape(-1, 1)
        v = Y_c[:, 1].reshape(-1, 1)
        P11 = Y_c[:, 2]
        P12 = Y_c[:, 3]
        P21 = Y_c[:, 4]
        P22 = Y_c[:, 5]
        p = Y_c[:, 6]
        phi = Y_c[:, 7]
        x_c = self.lambda_1
        y_c = self.lambda_2
        r = self.lambda_3
        # phi = torch.clamp(phi, 0, 1)
        phi_c = (-torch.tanh(
            (torch.sqrt(((self.X_c[:, 0] - x_c) ** 2 + (self.X_c[:, 1] - y_c) ** 2)) - r) / (
                    math.sqrt(2) * 0.003)) + 1) * 0.5
        phi_star = torch.zeros_like(phi).to(device)
        loss_phi = self.criterion(phi, phi_c)


        du_dX = torch.autograd.grad(inputs=self.X_c, outputs=u, grad_outputs=torch.ones_like(u), retain_graph=True,
                                    create_graph=True)[0]
        dv_dX = torch.autograd.grad(inputs=self.X_c, outputs=v, grad_outputs=torch.ones_like(v), retain_graph=True,
                                    create_graph=True)[0]
        gradU = torch.cat((du_dX, dv_dX), dim=1)
        size = gradU.size(0)
        gradU = gradU.reshape(size, 2, 2)
        I = torch.zeros_like(gradU).to(device)
        I[:, 0, 0] = 1
        I[:, 1, 1] = 1
        g_phi = (1 - phi) ** 2
        g_phi2 = torch.unsqueeze(g_phi.reshape(-1, 1), dim=-1)
        g_phi2 = torch.cat((g_phi2, g_phi2), dim=1)
        g_phi2 = torch.cat((g_phi2, g_phi2), dim=2)
        gradU = gradU * g_phi2
        F = gradU + I
        J = torch.det(F)
        J_star = torch.ones_like(J).to(device)
        loss_J = self.criterion(J, J_star)
        FT = torch.transpose(F, dim0=1, dim1=2)
        FTi = torch.inverse(FT)
        p = p.reshape(-1, 1)
        p = torch.unsqueeze(p, dim=-1)
        p = torch.cat((p, p), dim=1)
        p = torch.cat((p, p), dim=2)
        P = miu * F - p * FTi

        P11_star = P[:, 0, 0]
        P12_star = P[:, 0, 1]
        P21_star = P[:, 1, 0]
        P22_star = P[:, 1, 1]

        P11 = P11 * g_phi
        P12 = P12 * g_phi
        P21 = P21 * g_phi
        P22 = P22 * g_phi


        loss_P = self.criterion(P11, P11_star) + self.criterion(P12, P12_star) + self.criterion(P21, P21_star) + self.criterion(P22, P22_star)

        dP11_dx = \
            torch.autograd.grad(inputs=self.X_c, outputs=P11, grad_outputs=torch.ones_like(P11), retain_graph=True,
                                create_graph=True)[0][:, 0]
        dP12_dy = \
            torch.autograd.grad(inputs=self.X_c, outputs=P12, grad_outputs=torch.ones_like(P12), retain_graph=True,
                                create_graph=True)[0][:, 1]
        dP21_dx = \
            torch.autograd.grad(inputs=self.X_c, outputs=P21, grad_outputs=torch.ones_like(P21), retain_graph=True,
                                create_graph=True)[0][:, 0]
        dP22_dy = \
            torch.autograd.grad(inputs=self.X_c, outputs=P22, grad_outputs=torch.ones_like(P22), retain_graph=True,
                                create_graph=True)[0][:, 1]


        loss_pde1 = self.criterion(dP11_dx, - dP12_dy)
        loss_pde2 = self.criterion(dP21_dx, - dP22_dy)

        Y_in = self.model(self.X_data)
        u_in = Y_in[:, 0]
        v_in = Y_in[:, 1]
        phi_in = Y_in[:, 7]
        du_in_dX = \
            torch.autograd.grad(inputs=self.X_data, outputs=u_in, grad_outputs=torch.ones_like(u_in), retain_graph=True,
                                create_graph=True)[0]
        dv_in_dX = \
            torch.autograd.grad(inputs=self.X_data, outputs=v_in, grad_outputs=torch.ones_like(v_in), retain_graph=True,
                                create_graph=True)[0]
        gradU_in = torch.cat((du_in_dX, dv_in_dX), dim=1)
        size_in = gradU_in.size(0)
        gradU_in = gradU_in.reshape(size_in, 2, 2)
        I_in = torch.zeros_like(gradU_in).to(device)
        I_in[:, 0, 0] = 1
        I_in[:, 1, 1] = 1
        g_phi_in = (1 - phi_in) ** 2
        g_phi2_in = torch.unsqueeze(g_phi_in.reshape(-1, 1), dim=-1)
        g_phi2_in = torch.cat((g_phi2_in, g_phi2_in), dim=1)
        g_phi2_in = torch.cat((g_phi2_in, g_phi2_in), dim=2)
        gradU_in = gradU_in * g_phi2_in
        F_in = gradU_in + I_in
        # F_in = F[self.index,:]
        F11_in = F_in[:, 0, 0]
        F12_in = F_in[:, 0, 1]
        F21_in = F_in[:, 1, 0]
        F22_in = F_in[:, 1, 1]
        C11_in = F11_in * F11_in + F21_in * F21_in
        C12_in = F11_in * F12_in + F21_in * F22_in
        C22_in = F12_in * F12_in + F22_in * F22_in
        e11_in = 0.5 * (-1 + C11_in)
        e12_in = 0.5 * C12_in
        e22_in = 0.5 * (-1 + C22_in)
        loss_data = self.criterion(e11_in.reshape(-1, 1), self.e11_data) + 1 * self.criterion(e12_in.reshape(-1, 1),
                                                                                               self.e12_data) + self.criterion(
            e22_in.reshape(-1, 1), self.e22_data)

        loss_pde = loss_pde1 + loss_pde2
        loss_bc = loss_bd1 + loss_bd2 + loss_bd3 + loss_bd4 + loss_bd5 + loss_bd6 + loss_lb2
        loss_phi_bc = loss_phi1 + loss_phi2 + loss_phi3 + loss_phi4 + loss_phi5 + loss_phi6
        loss = 10 * loss_bc + loss_t + 3 * loss_P + loss_pde + 50 * loss_phi_bc + 50 * loss_data + 10 * loss_J + 10 * loss_phi
        loss.backward()


        if self.iter % 100 == 0:
            dcrease = -((loss.item() - self.loss_pre) / self.loss_pre) * 100
            self.loss_pre = loss.item()
            print(self.iter, loss.item(), f"dcrease{dcrease:.2f}%", 'CP Point:', self.X_c.size(0))
            print(f"loss_BC:{loss_bc.item()}")
            print(f"loss_t:{loss_t.item()}")
            print(f"loss_P:{loss_P.item()}")
            print(f"loss_pde:{loss_pde1.item() + loss_pde2.item()}")
            # print(f"loss_phi_bc:{loss_phi1.item() + loss_phi2.item() + loss_phi3.item() + loss_phi4.item()}")
            print(f"loss_phi:{loss_phi.item()}")
            print(f"loss_phi_bc:{loss_phi_bc.item()}")
            print(f"loss_J:{loss_J.item()}")
            print(f"loss_data:{loss_data.item()}")


        if self.iter % 1000 == 0:
            fig, ax = plt.subplots()
            ax.set_aspect('equal')
            X_c_current = self.X_c.cpu()
            X_c_current = X_c_current.detach().numpy()
            J_np = phi.cpu().detach().numpy()
            pm = ax.scatter(X_c_current[:, 0], X_c_current[:, 1], c=J_np, cmap='rainbow', marker='o', s=2, alpha=1, edgecolors='none')
            cbar = fig.colorbar(pm, ax=ax)
            ax.set_title('Iter:{}'.format(self.iter), fontsize=15)
            plt.pause(0.5)
            plt.savefig('fig/Hyperelastic_{}.png'.format(self.iter))
            plt.close()


        if self.iter % 10000 == 0:
            torch.save(net, 'Hyperelastic.pth')

        writer.add_scalar('loss', loss.item(), self.iter)
        writer.add_scalar('loss_pde', loss_pde.item(), self.iter)
        writer.add_scalar('loss_P', loss_P.item(), self.iter)
        writer.add_scalar('loss_J', loss_J.item(), self.iter)
        writer.add_scalar('loss_bc', loss_bc.item(), self.iter)
        writer.add_scalar('loss_phi', loss_phi.item(), self.iter)


        self.iter = self.iter + 1

        return loss


    def loss_func3(self):
        self.optimizer.zero_grad()
        Y_c = self.model(self.X_c)
        phi = Y_c[:, 7]
        x_c = self.lambda_1
        y_c = self.lambda_2
        r = self.lambda_3
        phi_c = (-torch.tanh(
            (torch.sqrt(((self.X_c[:, 0] - x_c) ** 2 + (self.X_c[:, 1] - y_c) ** 2)) - r) / (
                    math.sqrt(2) * 0.003)) + 1) * 0.5
        phi_c = torch.clamp(phi_c, 0, 1)
        loss_phi = self.criterion(phi, phi_c)

        loss = loss_phi

        loss.backward()

        if self.iter % 100 == 0:
            dcrease = -((loss.item() - self.loss_pre) / self.loss_pre) * 100
            self.loss_pre = loss.item()
            print(self.iter, loss.item(), f"dcrease{dcrease:.2f}%", 'CP Point:', self.X_c.size(0))

            print(f"loss_phi:{loss_phi.item()}")
            print(f"xc1:{x_c}")
            print(f"yc1:{y_c}")
            print(f"r1:{r}")




        if self.iter % 10000 == 0:
            torch.save(net, 'Hyperelastic.pth')

        writer.add_scalar('loss', loss.item(), self.iter)
        writer.add_scalar('xc', self.lambda_1, self.iter)
        writer.add_scalar('yc', self.lambda_2, self.iter)
        writer.add_scalar('r', self.lambda_3, self.iter)
        # writer.add_scalar('loss_pde', loss_pde.item(), self.iter)
        # writer.add_scalar('loss_f', loss_f.item(), self.iter)

        self.iter = self.iter + 1

        return loss


    def loss_func4(self):
        self.optimizer.zero_grad()
        # B = torch.randn(2, 256).cuda() * 10
        Y_1 = self.model(self.X_e1)
        P12_1 = Y_1[:, 3].reshape(-1, 1)
        P22_1 = Y_1[:, 5].reshape(-1, 1)
        phi_1 = Y_1[:, 7].reshape(-1, 1)
        loss_t = self.criterion(P22_1, self.t)
        loss_bd1 = self.criterion(P12_1, self.s1)
        loss_phi1 = self.criterion(phi_1, self.s1)

        Y_2 = self.model(self.X_e2)
        P11_2 = Y_2[:, 2].reshape(-1, 1)
        P21_2 = Y_2[:, 4].reshape(-1, 1)
        phi_2 = Y_2[:, 7].reshape(-1, 1)
        loss_bd2 = self.criterion(P11_2, self.s2) + self.criterion(P21_2, self.s2)
        loss_phi2 = self.criterion(phi_2, self.s2)

        Y_3 = self.model(self.X_e3)
        P11_3 = Y_3[:, 2].reshape(-1, 1)
        P21_3 = Y_3[:, 4].reshape(-1, 1)
        phi_3 = Y_3[:, 7].reshape(-1, 1)
        loss_bd3 = self.criterion(P11_3, self.s3) + self.criterion(P21_3, self.s3)
        loss_phi3 = self.criterion(phi_3, self.s3)

        Y_4 = self.model(self.X_e4)
        v_4 = Y_4[:, 1].reshape(-1, 1)
        P12_4 = Y_4[:, 3].reshape(-1, 1)
        P21_4 = Y_4[:, 4].reshape(-1, 1)
        phi_4 = Y_4[:, 7].reshape(-1, 1)
        loss_bd4 = self.criterion(v_4, self.s4) + self.criterion(P12_4, self.s4)
        loss_phi4 = self.criterion(phi_4, self.s4)

        Y_5 = self.model(self.X_e5)
        v_5 = Y_5[:, 1].reshape(-1, 1)
        P12_5 = Y_5[:, 3].reshape(-1, 1)
        P21_5 = Y_5[:, 4].reshape(-1, 1)
        phi_5 = Y_5[:, 7].reshape(-1, 1)
        loss_bd5 = self.criterion(v_5, self.s5) + self.criterion(P12_5, self.s5)
        loss_phi5 = self.criterion(phi_5, self.s5)

        Y_6 = self.model(self.X_e6)
        P11_6 = Y_6[:, 2].reshape(-1, 1)
        P12_6 = Y_6[:, 3].reshape(-1, 1)
        P21_6 = Y_6[:, 4].reshape(-1, 1)
        P22_6 = Y_6[:, 5].reshape(-1, 1)
        phi_6 = Y_6[:, 7].reshape(-1, 1)
        x_6 = self.X_e6[:, 0].reshape(-1, 1)
        y_6 = self.X_e6[:, 1].reshape(-1, 1)
        loss_phi6 = self.criterion(phi_6, self.s6)
        loss_bd6 = self.criterion(P11_6 * x_6 * (2 / 0.09), -P12_6 * (y_6 + 0.4) * (2 / 0.16)) + self.criterion(
            P21_6 * x_6 * (2 / 0.09), -P22_6 * (y_6 + 0.4) * (2 / 0.16))

        Y_lb2 = self.model(self.X_lb2)
        u_lb2 = Y_lb2[:, 0]
        v_lb2 = Y_lb2[:, 1]
        loss_lb2 = self.criterion(u_lb2, self.lb2) + self.criterion(v_lb2, self.lb2)

        Y_c = self.model(self.X_c)
        u = Y_c[:, 0].reshape(-1, 1)
        v = Y_c[:, 1].reshape(-1, 1)
        P11 = Y_c[:, 2]
        P12 = Y_c[:, 3]
        P21 = Y_c[:, 4]
        P22 = Y_c[:, 5]
        p = Y_c[:, 6]
        phi = Y_c[:, 7]
        x_c = self.lambda_1
        y_c = self.lambda_2
        r = self.lambda_3
        phi_c = (-torch.tanh(
            (torch.sqrt(((self.X_c[:, 0] - x_c) ** 2 + (self.X_c[:, 1] - y_c) ** 2)) - r) / (
                    math.sqrt(2) * 0.003)) + 1) * 0.5
        phi_c = torch.clamp(phi_c, 0, 1)
        loss_phi = self.criterion(phi, phi_c)


        du_dX = torch.autograd.grad(inputs=self.X_c, outputs=u, grad_outputs=torch.ones_like(u), retain_graph=True,
                                    create_graph=True)[0]
        dv_dX = torch.autograd.grad(inputs=self.X_c, outputs=v, grad_outputs=torch.ones_like(v), retain_graph=True,
                                    create_graph=True)[0]
        gradU = torch.cat((du_dX, dv_dX), dim=1)
        size = gradU.size(0)
        gradU = gradU.reshape(size, 2, 2)
        I = torch.zeros_like(gradU).to(device)
        I[:, 0, 0] = 1
        I[:, 1, 1] = 1
        g_phi = (1 - phi) ** 2
        g_phi2 = torch.unsqueeze(g_phi.reshape(-1, 1), dim=-1)
        g_phi2 = torch.cat((g_phi2, g_phi2), dim=1)
        g_phi2 = torch.cat((g_phi2, g_phi2), dim=2)
        gradU = gradU * g_phi2
        F = gradU + I
        J = torch.det(F)
        J_star = torch.ones_like(J).to(device)
        loss_J = self.criterion(J, J_star)
        FT = torch.transpose(F, dim0=1, dim1=2)
        FTi = torch.inverse(FT)
        p = p.reshape(-1, 1)
        p = torch.unsqueeze(p, dim=-1)
        p = torch.cat((p, p), dim=1)
        p = torch.cat((p, p), dim=2)
        P = miu * F - p * FTi

        P11_star = P[:, 0, 0]
        P12_star = P[:, 0, 1]
        P21_star = P[:, 1, 0]
        P22_star = P[:, 1, 1]

        P11 = P11 * g_phi
        P12 = P12 * g_phi
        P21 = P21 * g_phi
        P22 = P22 * g_phi


        loss_P = self.criterion(P11, P11_star) + self.criterion(P12, P12_star) + self.criterion(P21, P21_star) + self.criterion(P22, P22_star)

        dP11_dx = \
            torch.autograd.grad(inputs=self.X_c, outputs=P11, grad_outputs=torch.ones_like(P11), retain_graph=True,
                                create_graph=True)[0][:, 0]
        dP12_dy = \
            torch.autograd.grad(inputs=self.X_c, outputs=P12, grad_outputs=torch.ones_like(P12), retain_graph=True,
                                create_graph=True)[0][:, 1]
        dP21_dx = \
            torch.autograd.grad(inputs=self.X_c, outputs=P21, grad_outputs=torch.ones_like(P21), retain_graph=True,
                                create_graph=True)[0][:, 0]
        dP22_dy = \
            torch.autograd.grad(inputs=self.X_c, outputs=P22, grad_outputs=torch.ones_like(P22), retain_graph=True,
                                create_graph=True)[0][:, 1]


        loss_pde1 = self.criterion(dP11_dx, - dP12_dy)
        loss_pde2 = self.criterion(dP21_dx, - dP22_dy)

        Y_in = self.model(self.X_data)
        u_in = Y_in[:, 0]
        v_in = Y_in[:, 1]
        phi_in = Y_in[:, 7]
        du_in_dX = \
            torch.autograd.grad(inputs=self.X_data, outputs=u_in, grad_outputs=torch.ones_like(u_in), retain_graph=True,
                                create_graph=True)[0]
        dv_in_dX = \
            torch.autograd.grad(inputs=self.X_data, outputs=v_in, grad_outputs=torch.ones_like(v_in), retain_graph=True,
                                create_graph=True)[0]
        gradU_in = torch.cat((du_in_dX, dv_in_dX), dim=1)
        size_in = gradU_in.size(0)
        gradU_in = gradU_in.reshape(size_in, 2, 2)
        I_in = torch.zeros_like(gradU_in).to(device)
        I_in[:, 0, 0] = 1
        I_in[:, 1, 1] = 1
        g_phi_in = (1 - phi_in) ** 2
        g_phi2_in = torch.unsqueeze(g_phi_in.reshape(-1, 1), dim=-1)
        g_phi2_in = torch.cat((g_phi2_in, g_phi2_in), dim=1)
        g_phi2_in = torch.cat((g_phi2_in, g_phi2_in), dim=2)
        gradU_in = gradU_in * g_phi2_in
        F_in = gradU_in + I_in
        # F_in = F[self.index,:]
        F11_in = F_in[:, 0, 0]
        F12_in = F_in[:, 0, 1]
        F21_in = F_in[:, 1, 0]
        F22_in = F_in[:, 1, 1]
        C11_in = F11_in * F11_in + F21_in * F21_in
        C12_in = F11_in * F12_in + F21_in * F22_in
        C22_in = F12_in * F12_in + F22_in * F22_in
        e11_in = 0.5 * (-1 + C11_in)
        e12_in = 0.5 * C12_in
        e22_in = 0.5 * (-1 + C22_in)
        loss_data = self.criterion(e11_in.reshape(-1, 1), self.e11_data) + 1 * self.criterion(e12_in.reshape(-1, 1),
                                                                                               self.e12_data) + self.criterion(
            e22_in.reshape(-1, 1), self.e22_data)

        loss_pde = loss_pde1 + loss_pde2
        loss_bc = loss_bd1 + loss_bd2 + loss_bd3 + loss_bd4 + loss_bd5 + loss_bd6 + loss_lb2
        loss_phi_bc = loss_phi1 + loss_phi2 + loss_phi3 + loss_phi4 + loss_phi5 + loss_phi6
        loss = 10 * loss_bc + loss_t + 3 * loss_P + loss_pde + 50 * loss_phi_bc + 10 * loss_J + 10 * loss_phi
        loss.backward()


        if self.iter % 100 == 0:
            dcrease = -((loss.item() - self.loss_pre) / self.loss_pre) * 100
            self.loss_pre = loss.item()
            print(self.iter, loss.item(), f"dcrease{dcrease:.2f}%", 'CP Point:', self.X_c.size(0))
            print(f"loss_BC:{loss_bc.item()}")
            print(f"loss_t:{loss_t.item()}")
            print(f"loss_P:{loss_P.item()}")
            print(f"loss_pde:{loss_pde1.item() + loss_pde2.item()}")
            # print(f"loss_phi_bc:{loss_phi1.item() + loss_phi2.item() + loss_phi3.item() + loss_phi4.item()}")
            print(f"loss_phi:{loss_phi.item()}")
            print(f"loss_phi_bc:{loss_phi_bc.item()}")
            print(f"loss_J:{loss_J.item()}")
            print(f"loss_data:{loss_data.item()}")
            print(f"xc1:{x_c}")
            print(f"yc1:{y_c}")
            print(f"r1:{r}")


        if self.iter % 1000 == 0:
            fig, ax = plt.subplots()
            ax.set_aspect('equal')
            X_c_current = self.X_c.cpu()
            X_c_current = X_c_current.detach().numpy()
            J_np = phi.cpu().detach().numpy()
            pm = ax.scatter(X_c_current[:, 0], X_c_current[:, 1], c=J_np, cmap='rainbow', marker='o', s=2, alpha=1, edgecolors='none')
            cbar = fig.colorbar(pm, ax=ax)
            ax.set_title('Iter:{}'.format(self.iter), fontsize=15)
            plt.pause(0.5)
            plt.savefig('fig/Hyperelastic_{}.png'.format(self.iter))
            plt.close()


        if self.iter % 10000 == 0:
            torch.save(net, 'Hyperelastic.pth')

        writer.add_scalar('loss', loss.item(), self.iter)
        writer.add_scalar('loss_pde', loss_pde.item(), self.iter)
        writer.add_scalar('loss_P', loss_P.item(), self.iter)
        writer.add_scalar('loss_J', loss_J.item(), self.iter)
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
# net.model.load_state_dict(torch.load('Hyperelastic_param.pth'))
path = "fig"
if os.path.exists(path):
    shutil.rmtree(path)
else:
    pass
os.makedirs(path)


plt.ion()
net.train1()
torch.save(net, 'Hyperelasticpre.pth')
torch.save(net.model.state_dict(), 'Hyperelastic_pretrain_param.pth')
for para in net.model.net1.parameters():
    para.requires_grad = False
for para in net.model.net2.parameters():
    para.requires_grad = False
for para in net.model.net3.parameters():
    para.requires_grad = False
for para in net.model.net4.parameters():
    para.requires_grad = False
net.train3()
torch.save(net, 'Hyperelastic_phi1.pth')

for para in net.model.net4.parameters():
    para.requires_grad = True
net.lambda_1.requires_grad = False
net.lambda_2.requires_grad = False
net.lambda_3.requires_grad = False

for para in net.model.net1.parameters():
    para.requires_grad = True
for para in net.model.net2.parameters():
    para.requires_grad = True
for para in net.model.net3.parameters():
    para.requires_grad = True
net.train4()
torch.save(net, 'Hyperelastic_phi2.pth')
net.lambda_1.requires_grad = True
net.lambda_2.requires_grad = True
net.lambda_3.requires_grad = True
net.train2()
plt.ioff()
print('done')
exit()