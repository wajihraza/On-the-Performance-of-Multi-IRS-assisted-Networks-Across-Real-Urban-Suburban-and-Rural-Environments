import numpy as np
import matplotlib.pyplot as plt

N_val = [1024]
n_user_vals_2 = 350


rr = np.load('ab2.npy',allow_pickle=True)
rr= np.transpose(rr)
rr_350 = rr[59]



rr2 = np.load('ab1.npy',allow_pickle=True)
rr2= np.transpose(rr2)
rr2_350 = rr2[59]

def total_coverage_probability(rate_mat, threshold_rate):
    coverage_probability = np.sum(rate_mat > threshold_rate) / len(rate_mat)
    return coverage_probability

thresholds = np.arange(-20,0, 0.01)
coverage_probab_2 = []
coverage_probab_3 = []
coverage_prob = np.array([])

for threshold in thresholds:
    coverage_prob = np.append(coverage_prob, total_coverage_probability(rr_350, threshold))
coverage_probab_2 = np.append(coverage_probab_2, coverage_prob)
coverage_probab_2 = coverage_probab_2.reshape(len(N_val),-1)

coverage_prob = np.array([])

for threshold in thresholds:
    coverage_prob = np.append(coverage_prob, total_coverage_probability(rr2_350, threshold))
coverage_probab_3 = np.append(coverage_probab_3, coverage_prob)
coverage_probab_3 = coverage_probab_3.reshape(len(N_val),-1)

for i, N in enumerate(N_val):
  plt.plot(thresholds, coverage_probab_2[i], label="With IRS ")
  plt.plot(thresholds, coverage_probab_3[i], label="Without IRS")

plt.xlabel('SNR Threshold in dB')
plt.ylabel('SNR Coverage Probability')
plt.title('With or Without IRS for 350 users')
plt.legend()
plt.grid(True)


plt.savefig('IRSvsNOT.png', format='png')

plt.show()