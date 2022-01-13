import numpy as np
import node
import FusionCenter
import scipy.io as sio
import multiprocessing
import time

if __name__ == '__main__':

    t0 = time.time()
    N = 3  # 节点数
    atom_num = 20  # 字典原子数

    # 输入数据，给数据分块
    data = []
    # data_all = np.loadtxt('zc34600.csv', delimiter=',')
    # data_all = data_all.T
    data_all_mat = sio.loadmat('CSTHTAI.mat')
    data_all = data_all_mat['TrainData']
    data_part = data_all[:, 0:2000]
    data_part = data_part.astype(np.float)
    data_num = data_part.shape[1]
    node_data_num = int(data_num / N)  # 平均分配数据给各个节点
    for i in range(0, N):
        data.append(data_part[:, i * node_data_num:(i + 1) * node_data_num])

    # 每个节点字典的权重，暂时没考虑各个节点的权重
    weight = []
    for i in range(0, N):
        weight.append(1 / N)

    # 节点网络，目前不需要考虑网络结构
    # network = np.ones((N, N))
    # network = network - np.eye(N)
    # # network = np.zeros((N, N))
    #
    # # 创建节点间的管道列表
    # node_pipe = []
    # for i in range(0, N):
    #     node_pipe.append([])
    # for i in range(0, N):
    #     for j in range(i, N):
    #         if network[i, j]:
    #             p1, p2 = multiprocessing.Pipe()
    #             node_pipe[i].append(p1)
    #             node_pipe[j].append(p2)

    # 创建与fusion center的管道列表
    fusion_pipe_node = []
    fusion_pipe_fusion = []
    for i in range(0, N):
        p1, p2 = multiprocessing.Pipe()
        fusion_pipe_fusion.append(p1)
        fusion_pipe_node.append(p2)
    for m in range(0, 20):
        # 创建节点进程
        node_list = []
        for i in range(0, N):
            node_list.append(
                node.NODE(data=data[i], atom_num=atom_num, node_num=i, node_weight=weight[i],
                          fusion_pipe=fusion_pipe_node[i]))
            node_list[i].start()

        # 创建fusion center进程
        fusion_center = FusionCenter.FusionCenter(node_pipe=fusion_pipe_fusion, N=N, weight=weight, m=m)
        fusion_center.start()

        for i in range(0, N):
            node_list[i].join()
        fusion_center.join()
        print(m)
    print(time.time() - t0)
