import numpy as np
from numpy import linalg
import math

def roots(N):
    mat = np.zeros((N,N))
    mat[0,0] = 1
    mat[0,1] = -1
    mat[N-1,N-1] = 2*(N-1)+1
    mat[N-1,N-2] = -N+1
    for i in range(1,N-1):
        mat[i,i-1] = -i
        mat[i,i] = 2*i+1
        mat[i,i+1] = -(i+1)

    eigvalue, eigvector = np.linalg.eig(mat)
    return eigvalue


def Laguerre(N, x):
    x_num = x.shape[0]
    L_array = np.zeros((N + 1, x_num))
    Q_array = np.zeros((N + 1, x_num))

    for i in range(x_num):
        L_array[0][i] = 1
        L_array[1][i] = (2 * 0 + 1 - x[i]) * L_array[0][i] / (0 + 1) - 0 / (0 + 1) * 0

        for n in range(1, N):
            L_array[n + 1][i] = (2 * n + 1 - x[i]) * L_array[n][i] / (n + 1) - n / (n + 1) * L_array[n - 1][i]

    for i in range(x_num):
        Q_array[0][i] = 0
        Q_array[1][i] = 1

        for n in range(1, N):
            Q_array[n + 1][i] = L_array[n][i] + Q_array[n][i]

    return (L_array, Q_array)

def orthogonal_functions(N):
    x = roots(N)
    f,q = Laguerre(N,x)
    f_tem = f[:-1,:]
    f_final = np.zeros((N,N))
    for i in range(N):
        e = 1/(math.sqrt(x[i])*abs(q[-1,i]))
        f_final[:,i] = f_tem[:,i]*e
    return x,f_final


################
'''Problem 7'''
x ,f = orthogonal_functions(300)
print(np.max(np.abs(f@f.T-np.eye(len(x)))))