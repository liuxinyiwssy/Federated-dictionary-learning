import numpy as np
import scipy.io as sio


def wgn(x, snr):
    snr = 10 ** (snr / 10.0)
    xpower = np.sum(abs(x) ** 2) / len(x)
    npower = xpower / snr
    return np.random.randn(len(x)) * np.sqrt(npower)


generate_matrix = np.random.rand(20, 50) - 0.5
for k in range(0, 50):
    generate_matrix[:, k] = generate_matrix[:, k] / np.linalg.norm(generate_matrix[:, k], 2)

signal = []
for i in range(0, 3000):
    coefs = np.random.rand(4) - 0.5
    sign = generate_matrix[:, np.random.randint(0, 50, 4)]
    for j in range(0, 4):
        sign[:, j] = sign[:, j] * coefs[j]
    signal.append(np.sum(sign, axis=1).T)

signal = np.array(signal).T
noise_signal_10 = np.zeros((signal.shape))
noise_signal_20 = np.zeros((signal.shape))
noise_signal_30 = np.zeros((signal.shape))
noise_signal_40 = np.zeros((signal.shape))
noise_signal_50 = np.zeros((signal.shape))

for i in range(0, signal.shape[1]):
    noise_signal_10[:, i] = signal[:, i] + wgn(signal[:, i], 10)
    noise_signal_20[:, i] = signal[:, i] + wgn(signal[:, i], 20)
    noise_signal_30[:, i] = signal[:, i] + wgn(signal[:, i], 30)
    noise_signal_40[:, i] = signal[:, i] + wgn(signal[:, i], 40)
    noise_signal_50[:, i] = signal[:, i] + wgn(signal[:, i], 50)

sio.savemat('data.mat',
            {'signal': signal, 'noise_signal_10': noise_signal_10, 'noise_signal_20': noise_signal_20,
             'noise_signal_30': noise_signal_30, 'noise_signal_40': noise_signal_40,
             'noise_signal_50': noise_signal_50, 'generate': generate_matrix})
# np.savetxt('data.csv',signal,delimiter=',')
# np.savetxt('generate.csv',generate_matrix,delimiter=',')
