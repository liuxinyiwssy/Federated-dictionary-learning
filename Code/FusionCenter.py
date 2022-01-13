import multiprocessing
from scipy.optimize import linear_sum_assignment
import numpy as np
import scipy.io as sio
from OMP import omp
import time


class FusionCenter(multiprocessing.Process):

    def __init__(self, node_pipe, N, weight, m):
        multiprocessing.Process.__init__(self)
        self.NODE_pipe = node_pipe  # fusion center 和所有节点连接的管道
        self.dictionary = []  # 所有节点的字典列表
        self.transposed_matrix = None  # 所有节点的置换矩阵
        self.N = N  # 节点总数
        self.weight = weight  # 节点权重列表
        self.step = 0
        self.m = m

    def run(self) -> None:

        while True:
            self.dictionary = []
            for node_pipe in self.NODE_pipe:
                self.dictionary.append(node_pipe.recv())

            standard_dict = self.dictionary[0]  # 强制使用节点0的字典为标准
            atom_dim = standard_dict.shape[1]
            self.transposed_matrix = np.zeros((self.N, atom_dim, atom_dim))
            self.transposed_matrix[0, :, :] = np.eye(atom_dim)
            cost = np.zeros((self.N, atom_dim, atom_dim))
            Nn = list(range(0, self.N))
            Nn.remove(0)
            for n in Nn:
                for row in range(0, atom_dim):
                    for col in range(0, atom_dim):
                        cost[n, row, col] = pow(
                            np.linalg.norm((standard_dict[:, row] - self.dictionary[n][:, col]), 2), 2)

                row_ind, col_ind = linear_sum_assignment(cost[n, :, :])
                # 生成置换矩阵
                for k in range(0, atom_dim):
                    self.transposed_matrix[n, row_ind[k], col_ind[k]] = 1

            d = np.zeros(self.dictionary[0].shape)
            for i in range(0, self.N):
                d = d + np.dot(self.dictionary[i], self.transposed_matrix[i, :, :].T) * self.weight[i]
            for i in range(0, atom_dim):
                d[:, i] = d[:, i] / np.linalg.norm(d[:, i], 2)
            for node_pipe in self.NODE_pipe:
                node_pipe.send(d)
            self.step = self.step + 1
            if self.step == 5:
                sio.savemat('fusion' + str(self.m) + '.mat', {'D0': d})
                break
