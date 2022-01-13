import numpy as np
import scipy.io as sio
from OMP import omp
import time


class KSVD(object):
    def __init__(self, n_components, max_iter, n_nonzero_coefs, tol=1e-6):
        """
        稀疏模型Y = DX，Y为样本矩阵，使用KSVD动态更新字典矩阵D和稀疏矩阵X
        :param n_components: 字典所含原子个数（字典的列数）
        :param max_iter: 最大迭代次数
        :param tol: 稀疏表示结果的容差
        :param n_nonzero_coefs: 稀疏度
        """
        self.dictionary = None
        self.sparsecode = None
        self.max_iter = max_iter
        self.tol = tol
        self.n_components = n_components
        self.n_nonzero_coefs = n_nonzero_coefs
        self.dev = []

    def _update_dict(self, y, d, x):
        """
        使用KSVD更新字典的过程
        """
        for i in range(self.n_components):
            index = np.nonzero(x[i, :])[0]
            if len(index) == 0:
                continue
            d[:, i] = 0
            r = (y - np.dot(d, x))[:, index]
            u, s, v = np.linalg.svd(r, full_matrices=False)
            d[:, i] = u[:, 0].T
            x[i, index] = s[0] * v[0, :]
        return d

    def fit(self, y, d):
        """
        KSVD迭代过程
        """
        self.dictionary = d
        for i in range(self.max_iter):
            x = omp(self.dictionary, y, self.n_nonzero_coefs)
            self.dictionary = self._update_dict(y, self.dictionary, x)
            for k in range(0, self.n_components):
                self.dictionary[:, k] = self.dictionary[:, k] / np.linalg.norm(self.dictionary[:, k], 2)  # 标准化3
        return self.dictionary


if __name__ == '__main__':
    # data_all = np.loadtxt('zc34600.csv', delimiter=',')
    # data_all = data_all.T
    data_all_mat = sio.loadmat('CSTHTAI.mat')
    data_all = data_all_mat['TrainData']
    t0 = time.time()
    data_all = data_all.astype(np.float)
    data = data_all[:, 0:2000]
    ksvd = KSVD(n_components=20, n_nonzero_coefs=1, max_iter=30)
    for m in range(0, 20):
        dictionary = data[:, np.random.randint(0, 2000, 20)]
        for k in range(0, 20):
            dictionary[:, k] = dictionary[:, k] / np.linalg.norm(dictionary[:, k], 2)  # 标准化2
        dictionary = ksvd.fit(data, dictionary)
        print(time.time() - t0)
        print(m)
        sio.savemat('d' + str(m) + '.mat', {'D0': dictionary})
    # ksvd = KSVD(n_components=50, n_nonzero_coefs=1, max_iter=49)
    # dictionary = ksvd.fit(data, dictionary)