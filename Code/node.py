import numpy as np
from OMP import omp
import scipy.io as sio
import multiprocessing
from myksvd import KSVD
import time


class NODE(multiprocessing.Process):
    def __init__(self, data, atom_num, node_num, node_weight, fusion_pipe):
        multiprocessing.Process.__init__(self)
        self.node_num = node_num  # 节点编号
        self.data = data  # 节点数据
        self.data_dim = data.shape[0]  # 数据维数
        self.data_num = data.shape[1]  # 数据量
        self.atom_num = atom_num  # 字典原子数量
        self.exchange_step = 6  # KSVD最大迭代次数
        self.nonzero = 1  # OMP稀疏度
        self.weight = node_weight  # 当前节点的字典权重
        # 字典初始化
        self.dictionary = self.data[:, np.random.randint(0, self.data_num, self.atom_num)]
        for k in range(0, self.atom_num):
            self.dictionary[:, k] = self.dictionary[:, k] / np.linalg.norm(self.dictionary[:, k], 2)  # 标准化
        self.step = 0  # 迭代步数
        self.fusion_pipe = fusion_pipe  # fusion center 的管道

    def run(self) -> None:
        my_ksvd = KSVD(n_components=self.atom_num, n_nonzero_coefs=self.nonzero, max_iter=self.exchange_step)
        while True:
            self.dictionary = my_ksvd.fit(self.data, self.dictionary)
            self.exchange()
            self.step += 1
            if self.step == 5:
                break

    # 和fusion center 交换数据
    def exchange(self):
        # 向fusion center 发送本地字典
        self.fusion_pipe.send(self.dictionary)
        # 等待fusion center 发送字典
        self.dictionary = self.fusion_pipe.recv()



