import numpy as np
import pandas as pd
import math
import torch
import os
import random
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier


# functional similarity
def S_fun1(DDsim, T0, T1):
    DDsim = np.array(DDsim)
    T0_T1 = []
    if len(T0) != 0 and len(T1) != 0:
        for ti in T0:
            m_ax = []
            for tj in T1:
                m_ax.append(DDsim[ti][tj])
            T0_T1.append(max(m_ax))
    if len(T0) == 0 or len(T1) == 0:
        T0_T1.append(0)
    T1_T0 = []
    if len(T0) != 0 and len(T1) != 0:
        for tj in T1:
            m_ax = []
            for ti in T0:
                m_ax.append(DDsim[tj][ti])
            T1_T0.append(max(m_ax))
    if len(T0) == 0 or len(T1) == 0:
        T1_T0.append(0)
    return T0_T1, T1_T0

def FS_fun1(T0_T1, T1_T0, T0, T1):
    a = len(T1)
    b = len(T0)
    S1 = sum(T0_T1)
    S2 = sum(T1_T0)
    FS = []
    if a != 0 and b != 0:
        Fsim = (S1 + S2) / (a + b)
        FS.append(Fsim)
    if a == 0 or b == 0:
        FS.append(0)
    return FS



# Gaussian interaction profile kernel similarity
def r_func(MD):
    m = MD.shape[0]
    n = MD.shape[1]
    EUC_MD = np.linalg.norm(MD, ord=2, axis=1, keepdims=False)
    EUC_DL = np.linalg.norm(MD.T, ord=2, axis=1, keepdims=False)
    EUC_MD = EUC_MD ** 2
    EUC_DL = EUC_DL ** 2
    sum_EUC_MD = np.sum(EUC_MD)
    sum_EUC_DL = np.sum(EUC_DL)
    rl = 1 / ((1 / m) * sum_EUC_MD)
    rt = 1 / ((1 / n) * sum_EUC_DL)
    return rl, rt

def Gau_sim(MD, rl, rt):
    MD = np.mat(MD)
    DL = MD.T
    m = MD.shape[0]
    n = MD.shape[1]
    c = []
    d = []
    for i in range(m):
        for j in range(m):
            b_1 = MD[i] - MD[j]
            b_norm1 = np.linalg.norm(b_1, ord=None, axis=1, keepdims=False)
            b1 = b_norm1 ** 2
            b1 = math.exp(-rl * b1)
            c.append(b1)
    for i in range(n):
        for j in range(n):
            b_2 = DL[i] - DL[j]
            b_norm2 = np.linalg.norm(b_2, ord=None, axis=1, keepdims=False)
            b2 = b_norm2 ** 2
            b2 = math.exp(-rt * b2)
            d.append(b2)
    GMM = np.mat(c).reshape(m, m)
    GDD = np.mat(d).reshape(n, n)
    return GMM, GDD



# cosine similarity
def cos_sim(MD):
    m = MD.shape[0]
    n = MD.shape[1]
    cos_MS1 = []
    cos_DS1 = []
    for i in range(m):
        for j in range(m):
            a = MD.iloc[i, :]
            b = MD.iloc[j, :]
            a_norm = np.linalg.norm(a)
            b_norm = np.linalg.norm(b)
            if a_norm != 0 and b_norm != 0:
                cos_ms = np.dot(a, b) / (a_norm * b_norm)
                cos_MS1.append(cos_ms)
            else:
                cos_MS1.append(0)

    for i in range(n):
        for j in range(n):
            a1 = MD.iloc[:, i]
            b1 = MD.iloc[:, j]
            a1_norm = np.linalg.norm(a1)
            b1_norm = np.linalg.norm(b1)
            if a1_norm != 0 and b1_norm != 0:
                cos_ds = np.dot(a1, b1) / (a1_norm * b1_norm)
                cos_DS1.append(cos_ds)
            else:
                cos_DS1.append(0)

    cos_MS1 = np.array(cos_MS1).reshape(m, m)
    cos_DS1 = np.array(cos_DS1).reshape(n, n)
    return cos_MS1, cos_DS1


# sigmoid function kernel similarity
def sig_kr(MD):
    m = MD.shape[0]
    n = MD.shape[1]
    sig_MS1 = []
    sig_DS1 = []
    for i in range(m):
        for j in range(m):
            a = MD.iloc[i, :]
            b = MD.iloc[j, :]
            z = (1 / m) * (np.dot(a, b))
            sig_ms = math.tanh(z)
            sig_MS1.append(sig_ms)

    for i in range(n):
        for j in range(n):
            a1 = MD.iloc[:, i]
            b1 = MD.iloc[:, j]
            z1 = (1 / n) * (np.dot(a1, b1))
            sig_ds = math.tanh(z1)
            sig_DS1.append(sig_ds)

    sig_MS1 = np.array(sig_MS1).reshape(m, m)
    sig_DS1 = np.array(sig_DS1).reshape(n, n)
    return sig_MS1, sig_DS1


MD = pd.read_csv(r".data\adj_index.csv", index_col=0)

se_sim = np.loadtxt(r".\calc_similarity\miRNA\sequence_sim\seq_sim.txt")

DS = np.loadtxt(r".\calc_similarity\stress\semantic_sim\stress_sem_sim.txt")


# calculating functional similarity
m = MD.shape[0]
T = []
for i in range(m):
    T.append(np.where(MD.iloc[i] == 1))

Fs = []
for ti in range(m):
    for tj in range(m):
        Ti_Tj, Tj_Ti = S_fun1(DS, T[ti][0], T[tj][0])
        FS_i_j = FS_fun1(Ti_Tj, Tj_Ti, T[ti][0], T[tj][0])
        Fs.append(FS_i_j)

Fs = np.array(Fs).reshape(MD.shape[0], MD.shape[0])
Fs = pd.DataFrame(Fs)
for index, rows in Fs.iterrows():
    for col, rows in Fs.iteritems():
        if index == col:
            Fs.loc[index, col] = 1


rm, rt = r_func(MD)
GaM, GaD = Gau_sim(MD, rm, rt)

GaM = pd.DataFrame(GaM)
GaD = pd.DataFrame(GaD)

#cos_MS, cos_DS = cos_sim(MD)
#cos_MS = pd.DataFrame(cos_MS)
#cos_DS = pd.DataFrame(cos_DS)


#sig_MS, sig_DS = sig_kr(MD)
#sig_MS = pd.DataFrame(sig_MS)
#sig_DS = pd.DataFrame(sig_DS)


#Multi-source features fusion

MM = (se_sim + GaM + Fs) / 3
DD = (DS + GaD) / 2

MM = np.array(MM)
DD = np.array(DD)

# np.savetxt(r".\data\integration_sim\miRNA_sim_3.txt", MM)
# np.savetxt(r".\data\integration_sim\Stress_sim_2.txt", DD)

print(1)