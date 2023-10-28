import numpy as np
import pandas as pd


adj = np.loadtxt(r".\data\adj\adj_final.txt")
miRNA_sim = np.loadtxt(r".data\integration_sim\miRNA_sim_3.txt")
stress_sim = np.loadtxt(r".data\integration_sim\Stress_sim_2.txt")


adj1 = adj
for i in range(559):
    for j in range(559):
        adj1[i][j] = miRNA_sim[i][j]

for i in range(559, 614):
    for j in range(559, 614):
        adj1[i][j] = stress_sim[i-559][j-559]

#np.savetxt(r".data\integration_sim\heterogeneous_3_2.txt", adj1)

print(1)