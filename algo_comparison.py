import pickle
import matplotlib.pyplot as plt
import numpy as np

qsa_file_true = "results/Q_sa_true.pkl"
qsa_file_mc = "results/Q_sa_MC.pkl"
qsa_file_sarsa = "results/Q_sa_SARSA.pkl"
qsa_file_sarsa_approx = "results/Q_sa_sarsa_approx.pkl"
qsa_file_qlearning = "results/Q_sa_qlearning.pkl"

truth_results_dict = pickle.load(open(qsa_file_true, "rb"))
mc_results_dict = pickle.load(open(qsa_file_mc, "rb"))
sarsa_results_dict = pickle.load(open(qsa_file_sarsa, "rb"))
sarsaapprox_results_dict = pickle.load(open(qsa_file_sarsa_approx, "rb"))
qlearning_results_dict = pickle.load(open(qsa_file_qlearning, "rb"))

def cummean(x):
    return np.cumsum(x)/np.array(range(1, len(x)+1))

n_plot_episodes = 10000
# plot the mse per episode
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.plot(cummean(mc_results_dict["mc"]["reward_hist"]), label="MC")
ax1.plot(cummean(sarsa_results_dict["lambda"]["reward_hist"]), label="SARSA, lambda=0.2")
ax1.plot(cummean(sarsaapprox_results_dict["sarsa_approx"]["reward_hist"]), label="SARSA approx, lambda=0.2")
ax1.plot(cummean(qlearning_results_dict["qlearning"]["reward_hist"]), label="Q-learning")
# ax1.plot(cummean(performance_results_list[1][0]), label="b")
colormap = plt.cm.gist_ncar  # nipy_spectral, Set1,Paired
colors = [colormap(i) for i in np.linspace(0, 0.9, len(ax1.lines))]
for i, j in enumerate(ax1.lines):
    j.set_color(colors[i])
ax1.legend(loc=1)
ax1.set_xlabel('# Episodes')
ax1.set_ylabel('Avg. reward')
ax1.set_ylim([-1, 1])
ax1.set_xlim([0, n_plot_episodes])
plt.show()

# plot the average reward per episode
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.plot(mc_results_dict["mc"]["mse_hist"], label="MC")
ax1.plot(sarsa_results_dict["lambda"]["mse_hist"], label="SARSA, lambda=0.2")
ax1.plot(sarsaapprox_results_dict["sarsa_approx"]["mse_hist"], label="SARSA approx, lambda=0.2")
ax1.plot(qlearning_results_dict["qlearning"]["mse_hist"], label="Q-learning")
colormap = plt.cm.gist_ncar  # nipy_spectral, Set1,Paired
colors = [colormap(i) for i in np.linspace(0, 0.9, len(ax1.lines))]
for i, j in enumerate(ax1.lines):
    j.set_color(colors[i])
ax1.legend(loc=1)
ax1.set_xlabel('# Episodes')
ax1.set_ylabel('MSE')
ax1.set_xlim([0, n_plot_episodes])
plt.show()