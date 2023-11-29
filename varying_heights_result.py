import numpy as np
import matplotlib.pyplot as plt
n_user = 400

heights = [30,60,90,150,210]
def total_coverage_probability(rate_mat, threshold_rate):
    coverage_probability = np.sum(rate_mat > threshold_rate) / len(rate_mat)
    return coverage_probability

thresholds = np.arange(1.25e7, 2.8e7, 4000)
print(np.shape(thresholds))
rr2 = np.load('Code4_f_heights.npy',allow_pickle=True)
#rr2= np.transpose(rr2)


rr = np.load('Code4_f_2D_heights.npy',allow_pickle=True)
print(rr)

N_val = [256] 
coverage_probab_1 = []
for i in range(len(rr)):
  coverage_prob = np.array([])
  for threshold in thresholds:
      coverage_prob = np.append(coverage_prob, total_coverage_probability(rr[i], threshold))
  coverage_probab_1 = np.append(coverage_probab_1, coverage_prob)
coverage_probab_1 = coverage_probab_1.reshape(len(N_val),-1)

for i, N in enumerate(N_val):
  plt.plot(thresholds, coverage_probab_1[i], label="Height = 0 in 2D")



coverage_probab = []
coverage_probab_2 = []
coverage_probab_2 = np.zeros((rr2.shape[0], len(thresholds)))

for i in range(rr2.shape[0]):
    for j, threshold in enumerate(thresholds):
        coverage_probab_2[i, j] = total_coverage_probability(rr2[i, :], threshold)

for i, N in enumerate(rr2):
    plt.plot(thresholds, coverage_probab_2[i, :], label=f"Height = {heights[i]} in 3D")

#plt.plot(thresholds,  coverage_prob)
plt.xlabel('Threshold ')
plt.ylabel('Rate Coverage Probability')

plt.legend()
plt.grid(True)
plt.show()