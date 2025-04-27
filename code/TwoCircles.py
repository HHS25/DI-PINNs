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
writer = SummaryWriter('./path/to/log/TwoCircles')

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


def GenHoleSurfPT(xc, yc, r, N_PT):
    # Generate
    theta = lhs(1, N_PT)
    theta = theta * np.pi/2
    xx = np.multiply(r, np.cos(theta)) + xc
    yy = np.multiply(r, np.sin(theta)) + yc
    xx = xx.flatten()[:, None]
    yy = yy.flatten()[:, None]
    return xx, yy


def DelHolePT(XYT_c, xc=0, yc=0, r=0.1):
    # Delete points within hole
    dst = np.array([((xyt[0] - xc) ** 2 + (xyt[1] - yc) ** 2) ** 0.5 for xyt in XYT_c])
    return XYT_c[dst > r, :]


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
        data_c = DelHolePT(XYT_c=data_c, xc=-0.7, yc=0.0, r=0.5)
        data_c = DelHolePT(XYT_c=data_c, xc=0.7, yc=0.0, r=0.5)
        X_c = data_c

        a = np.arcsin(0.8)
        b = np.cos(math.pi / 3)
        theta = lhs(1, 200)
        theta = theta * 2 * a
        edge1_x = -0.4 + 0.8 * lhs(1, 200)
        edge1_y = 0.4 * np.ones_like(edge1_x)
        edge2_x = 0.5 * np.cos(theta - a) - 0.7
        edge2_y = 0.5 * np.sin(theta - a)
        edge3_x = 0.5 * np.cos(theta - a + math.pi) + 0.7
        edge3_y = 0.5 * np.sin(theta - a + math.pi)
        edge4_x = -0.4 + 0.8 * lhs(1, 200)
        edge4_y = -0.4 * np.ones_like(edge4_x)

        X_e1 = np.hstack((edge1_x, edge1_y))
        X_e2 = np.hstack((edge2_x, edge2_y))
        X_e3 = np.hstack((edge3_x, edge3_y))
        X_e4 = np.hstack((edge4_x, edge4_y))

        s1 = np.zeros((200, 1))
        s2 = np.zeros((200, 1))
        s3 = np.zeros((200, 1))
        s4 = np.zeros((200, 1))
        t1 = (-25 / 4) * edge1_x * edge1_x + 1
        t2 = 1

        X_lb2 = torch.tensor((-0.4, -0.4)).reshape(1, 2)
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
        t1 = torch.tensor(t1)
        self.t1 = t1
        self.t1 = self.t1.to(device)
        t2 = torch.tensor(t2)
        self.t2 = t2
        self.t2 = self.t2.to(device)

        # measurement points data
        path = 'TwoCircles.csv'
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


        ### Geometric parameters ###
        a1 = np.random.rand() * 0.16 - 0.24
        b1 = np.random.rand() * 0.16 + 0.1
        self.lambda_1 = torch.tensor([0.08], requires_grad=True).to(device)
        self.lambda_2 = torch.tensor([0.15], requires_grad=True).to(device)
        self.lambda_3 = torch.tensor([0.05], requires_grad=True).to(device)
        self.lambda_1 = torch.nn.Parameter(self.lambda_1)
        self.lambda_2 = torch.nn.Parameter(self.lambda_2)
        self.lambda_3 = torch.nn.Parameter(self.lambda_3)
        self.model.register_parameter('lambda_1', self.lambda_1)
        self.model.register_parameter('lambda_2', self.lambda_2)
        self.model.register_parameter('lambda_3', self.lambda_3)
        a2 = np.random.rand() * 0.16 - 0.24
        b2 = np.random.rand() * 0.16 + 0.1
        self.lambda_4 = torch.tensor([-0.1], requires_grad=True).to(device)
        self.lambda_5 = torch.tensor([-0.25], requires_grad=True).to(device)
        self.lambda_6 = torch.tensor([0.05], requires_grad=True).to(device)
        self.lambda_4 = torch.nn.Parameter(self.lambda_4)
        self.lambda_5 = torch.nn.Parameter(self.lambda_5)
        self.lambda_6 = torch.nn.Parameter(self.lambda_6)
        self.model.register_parameter('lambda_4', self.lambda_4)
        self.model.register_parameter('lambda_5', self.lambda_5)
        self.model.register_parameter('lambda_6', self.lambda_6)

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

        Y_1 = self.model(self.X_e1)
        s12_1 = Y_1[:, 3].reshape(-1, 1)
        s22_1 = Y_1[:, 4].reshape(-1, 1)
        phi_1 = Y_1[:, 5].reshape(-1, 1)
        loss_phi_1 = self.criterion(phi_1, self.s1)
        loss_bd1 = self.criterion(s12_1, self.s1)
        loss_t1 = self.criterion(s22_1, self.t1)

        Y_2 = self.model(self.X_e2)
        s11_2 = Y_2[:, 2].reshape(-1, 1)
        s12_2 = Y_2[:, 3].reshape(-1, 1)
        s22_2 = Y_2[:, 4].reshape(-1, 1)
        phi_2 = Y_2[:, 5].reshape(-1, 1)
        x_2 = self.X_e2[:, 0].reshape(-1, 1)
        y_2 = self.X_e2[:, 1].reshape(-1, 1)
        loss_phi_2 = self.criterion(phi_2, self.s2)
        loss_bd2 = self.criterion(s11_2 * (x_2 + 0.7), -s12_2 * y_2) + self.criterion(s12_2 * (x_2 + 0.7), -s22_2 * y_2)

        Y_3 = self.model(self.X_e3)
        s11_3 = Y_3[:, 2].reshape(-1, 1)
        s12_3 = Y_3[:, 3].reshape(-1, 1)
        s22_3 = Y_3[:, 4].reshape(-1, 1)
        phi_3 = Y_3[:, 5].reshape(-1, 1)
        x_3 = self.X_e3[:, 0].reshape(-1, 1)
        y_3 = self.X_e3[:, 1].reshape(-1, 1)
        loss_phi_3 = self.criterion(phi_3, self.s3)
        loss_bd3 = self.criterion(s11_3 * (x_3 - 0.7), -s12_3 * y_3) + self.criterion(s12_3 * (x_3 - 0.7), -s22_3 * y_3)

        Y_4 = self.model(self.X_e4)
        s12_4 = Y_4[:, 3].reshape(-1, 1)
        v_4 = Y_4[:, 1].reshape(-1, 1)
        phi_4 = Y_4[:, 5].reshape(-1, 1)
        loss_phi_4 = self.criterion(phi_4, self.s4)
        loss_bd4 = self.criterion(s12_4, self.s4) + self.criterion(v_4, self.s4)

        Y_lb2 = self.model(self.X_lb2)
        u_lb2 = Y_lb2[:, 0]
        v_lb2 = Y_lb2[:, 1]
        loss_lb2 = self.criterion(u_lb2, self.lb2) + self.criterion(v_lb2, self.lb2)

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
        loss_bc = loss_bd1 + loss_bd2 + loss_bd3 + loss_bd4 + loss_lb2
        loss_f = loss_f1 + loss_f2 + loss_f3
        loss_phi_bc = loss_phi_1 + loss_phi_2 + loss_phi_3 + loss_phi_4
        loss_t = loss_t1

        loss = 10 * loss_bc + 1 * loss_t + 3 * loss_f + loss_pde + 10 * loss_data + 10 * loss_phi_bc
        loss.backward()

        ### plot ###
        if self.iter % 1000 == 0:
            fig, ax = plt.subplots()
            ax.set_aspect('equal')
            X_c_current = self.X_c.cpu()
            X_c_current = X_c_current.detach().numpy()
            J_np = phi.cpu().detach().numpy()
            pm = ax.scatter(X_c_current[:, 0], X_c_current[:, 1], c=J_np, cmap='rainbow', marker='o', s=2, alpha=1,
                            edgecolors='none')
            cbar = fig.colorbar(pm, ax=ax)
            ax.set_title('Iter:{}'.format(self.iter), fontsize=15)
            plt.pause(0.5)
            plt.savefig('fig/TwoCircles_{}.png'.format(self.iter))
            plt.close()

        if self.iter % 100 == 0:
            dcrease = -((loss.item() - self.loss_pre) / self.loss_pre) * 100
            self.loss_pre = loss.item()
            print(self.iter, loss.item(), f"dcrease{dcrease:.2f}%", 'CP Point:', self.X_c.size(0))
            print(f"loss_BC:{loss_bc.item()}")
            print(f"loss_t:{loss_t.item()}")
            print(f"loss_f:{loss_f.item()}")
            print(f"loss_pde:{loss_pde1.item() + loss_pde2.item()}")
            print(f"loss_phi_bc:{loss_phi_bc.item()}")
            print(f"loss_data:{loss_data.item()}")

        if self.iter % 10000 == 0:
            torch.save(net, 'TwoCircles.pth')

        writer.add_scalar('loss', loss.item(), self.iter)
        writer.add_scalar('loss_pde', loss_pde.item(), self.iter)
        writer.add_scalar('loss_f', loss_f.item(), self.iter)
        writer.add_scalar('loss_bc', loss_bc.item(), self.iter)
        writer.add_scalar('loss_t', loss_t.item(), self.iter)
        writer.add_scalar('loss_phi', loss_phi.item(), self.iter)
        writer.add_scalar('loss_phi_bc', loss_phi_bc.item(), self.iter)
        writer.add_scalar('loss_data', loss_data.item(), self.iter)

        self.iter = self.iter + 1

        return loss

    def loss_func2(self):
        self.optimizer.zero_grad()

        Y_1 = self.model(self.X_e1)
        s12_1 = Y_1[:, 3].reshape(-1, 1)
        s22_1 = Y_1[:, 4].reshape(-1, 1)
        phi_1 = Y_1[:, 5].reshape(-1, 1)
        loss_phi_1 = self.criterion(phi_1, self.s1)
        loss_bd1 = self.criterion(s12_1, self.s1)
        loss_t1 = self.criterion(s22_1, self.t1)

        Y_2 = self.model(self.X_e2)
        s11_2 = Y_2[:, 2].reshape(-1, 1)
        s12_2 = Y_2[:, 3].reshape(-1, 1)
        s22_2 = Y_2[:, 4].reshape(-1, 1)
        phi_2 = Y_2[:, 5].reshape(-1, 1)
        x_2 = self.X_e2[:, 0].reshape(-1, 1)
        y_2 = self.X_e2[:, 1].reshape(-1, 1)
        loss_phi_2 = self.criterion(phi_2, self.s2)
        loss_bd2 = self.criterion(s11_2 * (x_2 + 0.7), -s12_2 * y_2) + self.criterion(s12_2 * (x_2 + 0.7), -s22_2 * y_2)

        Y_3 = self.model(self.X_e3)
        s11_3 = Y_3[:, 2].reshape(-1, 1)
        s12_3 = Y_3[:, 3].reshape(-1, 1)
        s22_3 = Y_3[:, 4].reshape(-1, 1)
        phi_3 = Y_3[:, 5].reshape(-1, 1)
        x_3 = self.X_e3[:, 0].reshape(-1, 1)
        y_3 = self.X_e3[:, 1].reshape(-1, 1)
        loss_phi_3 = self.criterion(phi_3, self.s3)
        loss_bd3 = self.criterion(s11_3 * (x_3 - 0.7), -s12_3 * y_3) + self.criterion(s12_3 * (x_3 - 0.7), -s22_3 * y_3)

        Y_4 = self.model(self.X_e4)
        s12_4 = Y_4[:, 3].reshape(-1, 1)
        v_4 = Y_4[:, 1].reshape(-1, 1)
        phi_4 = Y_4[:, 5].reshape(-1, 1)
        loss_phi_4 = self.criterion(phi_4, self.s4)
        loss_bd4 = self.criterion(s12_4, self.s4) + self.criterion(v_4, self.s4)

        Y_lb2 = self.model(self.X_lb2)
        u_lb2 = Y_lb2[:, 0]
        v_lb2 = Y_lb2[:, 1]
        loss_lb2 = self.criterion(u_lb2, self.lb2) + self.criterion(v_lb2, self.lb2)

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
        r1 = self.lambda_3
        x_c2 = self.lambda_4
        y_c2 = self.lambda_5
        r2 = self.lambda_6
        phi_c1 = (-torch.tanh(
            (torch.sqrt(((self.X_c[:, 0] - x_c1) ** 2 + (self.X_c[:, 1] - y_c1) ** 2)) - r1) / (
                    math.sqrt(2) * 0.003)) + 1) * 0.5
        phi_c2 = (-torch.tanh(
            (torch.sqrt(((self.X_c[:, 0] - x_c2) ** 2 + (self.X_c[:, 1] - y_c2) ** 2)) - r2) / (
                    math.sqrt(2) * 0.003)) + 1) * 0.5
        phi_c = phi_c1 + phi_c2
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
        loss_bc = loss_bd1 + loss_bd2 + loss_bd3 + loss_bd4 + loss_lb2
        loss_f = loss_f1 + loss_f2 + loss_f3
        loss_phi_bc = loss_phi_1 + loss_phi_2 + loss_phi_3 + loss_phi_4
        loss_t = loss_t1

        loss = 10 * loss_bc + 1 * loss_t + 3 * loss_f + loss_pde + 30 * loss_data + 10 * loss_phi_bc + 10 * loss_phi
        loss.backward()

        if self.iter % 100 == 0:
            dcrease = -((loss.item() - self.loss_pre) / self.loss_pre) * 100
            self.loss_pre = loss.item()
            print(self.iter, loss.item(), f"dcrease{dcrease:.2f}%", 'CP Point:', self.X_c.size(0))
            print(f"loss_BC:{loss_bc.item()}")
            print(f"loss_t:{loss_t.item()}")
            print(f"loss_f:{loss_f.item()}")
            print(f"loss_pde:{loss_pde1.item() + loss_pde2.item()}")
            print(f"loss_phi_bc:{loss_phi_bc.item()}")
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
            plt.savefig('fig/TwoCircles_{}.png'.format(self.iter))
            plt.close()

        if self.iter % 10000 == 0:
            torch.save(net, 'TwoCircles.pth')

        writer.add_scalar('loss', loss.item(), self.iter)
        writer.add_scalar('loss_pde', loss_pde.item(), self.iter)
        writer.add_scalar('loss_f', loss_f.item(), self.iter)
        writer.add_scalar('loss_bc', loss_bc.item(), self.iter)
        writer.add_scalar('loss_phi', loss_phi.item(), self.iter)
        writer.add_scalar('loss_data', loss_data.item(), self.iter)
        writer.add_scalar('xc', self.lambda_1, self.iter)
        writer.add_scalar('yc', self.lambda_2, self.iter)
        writer.add_scalar('r', self.lambda_3, self.iter)

        self.iter = self.iter + 1

        return loss

    def loss_func3(self):
        self.optimizer.zero_grad()
        Y_c = self.model(self.X_c)

        phi = Y_c[:, 5]
        x_c1 = self.lambda_1
        y_c1 = self.lambda_2
        r1 = self.lambda_3
        x_c2 = self.lambda_4
        y_c2 = self.lambda_5
        r2 = self.lambda_6
        phi_c1 = (-torch.tanh(
            (torch.sqrt(((self.X_c[:, 0] - x_c1) ** 2 + (self.X_c[:, 1] - y_c1) ** 2)) - r1) / (
                    math.sqrt(2) * 0.003)) + 1) * 0.5
        phi_c2 = (-torch.tanh(
            (torch.sqrt(((self.X_c[:, 0] - x_c2) ** 2 + (self.X_c[:, 1] - y_c2) ** 2)) - r2) / (
                    math.sqrt(2) * 0.003)) + 1) * 0.5 * 0.5
        phi_c = phi_c1 + phi_c2
        phi_c = torch.clamp(phi_c, 0, 1)
        loss_phi = self.criterion(phi, phi_c)

        loss = 10 * loss_phi

        loss.backward()

        if self.iter % 100 == 0:
            dcrease = -((loss.item() - self.loss_pre) / self.loss_pre) * 100
            self.loss_pre = loss.item()
            print(self.iter, loss.item(), f"dcrease{dcrease:.2f}%", 'CP Point:', self.X_c.size(0))
            print(f"loss_phi:{loss_phi.item()}")
            print(f"xc1:{x_c1}")
            print(f"yc1:{y_c1}")
            print(f"r1:{r1}")
            print(f"xc2:{x_c2}")
            print(f"yc2:{y_c2}")
            print(f"r2:{r2}")

        if self.iter % 10000 == 0:
            torch.save(net, 'TwoCircles.pth')

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
            plt.savefig('fig/TwoCircles_{}.png'.format(self.iter))
            plt.close()

        writer.add_scalar('loss', loss.item(), self.iter)

        self.iter = self.iter + 1

        return loss

    def loss_func4(self):
        self.optimizer.zero_grad()

        Y_1 = self.model(self.X_e1)
        s12_1 = Y_1[:, 3].reshape(-1, 1)
        s22_1 = Y_1[:, 4].reshape(-1, 1)
        phi_1 = Y_1[:, 5].reshape(-1, 1)
        loss_phi_1 = self.criterion(phi_1, self.s1)
        loss_bd1 = self.criterion(s12_1, self.s1)
        loss_t1 = self.criterion(s22_1, self.t1)

        Y_2 = self.model(self.X_e2)
        s11_2 = Y_2[:, 2].reshape(-1, 1)
        s12_2 = Y_2[:, 3].reshape(-1, 1)
        s22_2 = Y_2[:, 4].reshape(-1, 1)
        phi_2 = Y_2[:, 5].reshape(-1, 1)
        x_2 = self.X_e2[:, 0].reshape(-1, 1)
        y_2 = self.X_e2[:, 1].reshape(-1, 1)
        loss_phi_2 = self.criterion(phi_2, self.s2)
        loss_bd2 = self.criterion(s11_2 * (x_2 + 0.7), -s12_2 * y_2) + self.criterion(s12_2 * (x_2 + 0.7), -s22_2 * y_2)

        Y_3 = self.model(self.X_e3)
        s11_3 = Y_3[:, 2].reshape(-1, 1)
        s12_3 = Y_3[:, 3].reshape(-1, 1)
        s22_3 = Y_3[:, 4].reshape(-1, 1)
        phi_3 = Y_3[:, 5].reshape(-1, 1)
        x_3 = self.X_e3[:, 0].reshape(-1, 1)
        y_3 = self.X_e3[:, 1].reshape(-1, 1)
        loss_phi_3 = self.criterion(phi_3, self.s3)
        loss_bd3 = self.criterion(s11_3 * (x_3 - 0.7), -s12_3 * y_3) + self.criterion(s12_3 * (x_3 - 0.7), -s22_3 * y_3)

        Y_4 = self.model(self.X_e4)
        s12_4 = Y_4[:, 3].reshape(-1, 1)
        v_4 = Y_4[:, 1].reshape(-1, 1)
        phi_4 = Y_4[:, 5].reshape(-1, 1)
        loss_phi_4 = self.criterion(phi_4, self.s4)
        loss_bd4 = self.criterion(s12_4, self.s4) + self.criterion(v_4, self.s4)

        Y_lb2 = self.model(self.X_lb2)
        u_lb2 = Y_lb2[:, 0]
        v_lb2 = Y_lb2[:, 1]
        loss_lb2 = self.criterion(u_lb2, self.lb2) + self.criterion(v_lb2, self.lb2)

        Y_c = self.model(self.X_c)
        u = Y_c[:, 0].reshape(-1, 1)
        v = Y_c[:, 1].reshape(-1, 1)
        s11 = Y_c[:, 2]
        s12 = Y_c[:, 3]
        s22 = Y_c[:, 4]
        phi = Y_c[:, 5]
        x_c1 = self.lambda_1
        y_c1 = self.lambda_2
        r1 = self.lambda_3
        x_c2 = self.lambda_4
        y_c2 = self.lambda_5
        r2 = self.lambda_6
        phi_c1 = (-torch.tanh(
            (torch.sqrt(((self.X_c[:, 0] - x_c1) ** 2 + (self.X_c[:, 1] - y_c1) ** 2)) - r1) / (
                    math.sqrt(2) * 0.003)) + 1) * 0.5
        phi_c2 = (-torch.tanh(
            (torch.sqrt(((self.X_c[:, 0] - x_c2) ** 2 + (self.X_c[:, 1] - y_c2) ** 2)) - r2) / (
                    math.sqrt(2) * 0.003)) + 1) * 0.5
        phi_c = phi_c1 + phi_c2
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
        loss_bc = loss_bd1 + loss_bd2 + loss_bd3 + loss_bd4 + loss_lb2
        loss_f = loss_f1 + loss_f2 + loss_f3
        loss_phi_bc = loss_phi_1 + loss_phi_2 + loss_phi_3 + loss_phi_4
        loss_t = loss_t1

        loss = 10 * loss_bc + 1 * loss_t + 3 * loss_f + loss_pde + 10 * loss_phi + 20 * loss_phi_bc
        loss.backward()

        if self.iter % 100 == 0:
            dcrease = -((loss.item() - self.loss_pre) / self.loss_pre) * 100
            self.loss_pre = loss.item()
            print(self.iter, loss.item(), f"dcrease{dcrease:.2f}%", 'CP Point:', self.X_c.size(0))
            print(f"loss_BC:{loss_bc.item()}")
            print(f"loss_t:{loss_t.item()}")
            print(f"loss_f:{loss_f.item()}")
            print(f"loss_pde:{loss_pde1.item() + loss_pde2.item()}")
            print(f"loss_phi_bc:{loss_phi_bc.item()}")
            print(f"loss_phi:{loss_phi.item()}")
            # print(f"loss_data:{loss_data.item()}")

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
            plt.savefig('fig/TwoCircles_{}.png'.format(self.iter))
            plt.close()

        if self.iter % 10000 == 0:
            torch.save(net, 'TwoCircles.pth')

        writer.add_scalar('loss', loss.item(), self.iter)
        writer.add_scalar('loss_pde', loss_pde.item(), self.iter)
        writer.add_scalar('loss_f', loss_f.item(), self.iter)
        writer.add_scalar('loss_bc', loss_bc.item(), self.iter)
        writer.add_scalar('loss_t', loss_t.item(), self.iter)
        writer.add_scalar('loss_phi', loss_phi.item(), self.iter)
        writer.add_scalar('loss_phi_bc', loss_phi_bc.item(), self.iter)

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
# net.model.load_state_dict(torch.load('TwoCircles_pretrain_param.pth'))

path = "fig"
if os.path.exists(path):
    shutil.rmtree(path)
else:
    pass
os.makedirs(path)

plt.ion()
net.train1()
torch.save(net, 'Step1.pth')
torch.save(net.model.state_dict(), 'TwoCircles_pretrain_param.pth')
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