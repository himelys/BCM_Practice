import pymc3 as pm
import numpy as np
import theano.tensor as tt
from IPython.core.pylabtools import figsize
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats.mstats import mquantiles

# N = 100
# with pm.Model() as model:
#     p = pm.Uniform("freq_cheating", 0, 1)
#
# with model:
#     true_answers = pm.Bernoulli("truths", p, shape=N, testval=np.random.binomial(1, 0.5, N))
#
# with model:
#     first_coin_flips = pm.Bernoulli("first_flips", 0.5, shape=N, testval=np.random.binomial(1, 0.5, N))
# print(first_coin_flips.tag.test_value)
#
# with model:
#     second_coin_flips = pm.Bernoulli("second_flips", 0.5, shape=N, testval=np.random.binomial(1, 0.5, N))
#
# with model:
#     val = first_coin_flips*true_answers + (1 - first_coin_flips)*second_coin_flips
#     observed_proportion = pm.Deterministic("observed_proportion", tt.sum(val)/float(N))
#
# print(observed_proportion.tag.test_value)
#
# X = 35
#
# with model:
#     observations = pm.Binomial("obs", N, observed_proportion, observed=X)
#
# # To be explained in Chapter 3!
# with model:
#     step = pm.Metropolis(vars=[p])
#     trace = pm.sample(40000, step=step)
#     burned_trace = trace[15000:]
#
# figsize(12.5, 3)
# p_trace = burned_trace["freq_cheating"][15000:]
# plt.hist(p_trace, histtype="stepfilled", density=True, alpha=0.85, bins=30,
#          label="posterior distribution", color="#348ABD")
# plt.vlines([.05, .35], [0, 0], [5, 5], alpha=0.3)
# plt.xlim(0, 1)
# plt.legend()

# Example: Challenger Space Shuttle Disaster
challenger_data = np.genfromtxt("data/challenger_data.csv", skip_header=1,
                                usecols=[1, 2], missing_values="NA",
                                delimiter=",")
# figsize(12.5, 3.5)
# np.set_printoptions(precision=3, suppress=True)
# #drop the NA values
# challenger_data = challenger_data[~np.isnan(challenger_data[:, 1])]
#
# #plot it, as a function of tempature (the first column)
# print("Temp (F), O-Ring failure?")
# # print(challenger_data)
#
# plt.scatter(challenger_data[:, 0], challenger_data[:, 1], s=75, color="k",
#             alpha=0.5)
# plt.yticks([0, 1])
# plt.ylabel("Damage Incident?")
# plt.xlabel("Outside temperature (Fahrenheit)")
# plt.title("Defects of the Space Shuttle O-Rings vs temperature");

# figsize(12, 3)
#
# def logistic(x, beta):
#     return 1.0 / (1.0 + np.exp(beta * x))
#
# x = np.linspace(-4, 4, 100)
# plt.plot(x, logistic(x, 1), label=r"$\beta = 1$")
# plt.plot(x, logistic(x, 3), label=r"$\beta = 3$")
# plt.plot(x, logistic(x, -5), label=r"$\beta = -5$")
# plt.legend()
# def logistic(x, beta, alpha=0):
#     return 1.0 / (1.0 + np.exp(np.dot(beta, x) + alpha))
#
# x = np.linspace(-4, 4, 100)
#
# plt.plot(x, logistic(x, 1), label=r"$\beta = 1$", ls="--", lw=1)
# plt.plot(x, logistic(x, 3), label=r"$\beta = 3$", ls="--", lw=1)
# plt.plot(x, logistic(x, -5), label=r"$\beta = -5$", ls="--", lw=1)
#
# plt.plot(x, logistic(x, 1, 1), label=r"$\beta = 1, \alpha = 1$",
#          color="#348ABD")
# plt.plot(x, logistic(x, 3, -2), label=r"$\beta = 3, \alpha = -2$",
#          color="#A60628")
# plt.plot(x, logistic(x, -5, 7), label=r"$\beta = -5, \alpha = 7$",
#          color="#7A68A6")
#
# plt.legend(loc="lower left")

temperature = challenger_data[:, 0]
D = challenger_data[:, 1]  # defect or not?

#notice the`value` here. We explain why below.
with pm.Model() as model:
    beta = pm.Normal("beta", mu=0.2, tau=0.001, testval=0)
    alpha = pm.Normal("alpha", mu=-10, tau=0.001, testval=0)
    p = pm.Deterministic("p", 1.0/(1. + tt.exp(beta*temperature + alpha)))

# connect the probabilities in `p` with our observations through a
# Bernoulli random variable.
with model:
    observed = pm.Bernoulli("bernoulli_obs", p, observed=D)

    # Mysterious code to be explained in Chapter 3
    start = pm.find_MAP()
    step = pm.Metropolis()
    trace = pm.sample(120000, step=step, start=start)
    burned_trace = trace[100000::2]

alpha_samples = burned_trace["alpha"][:, None]  # best to make them 1d
beta_samples = burned_trace["beta"][:, None]

# plt.figure(1)
figsize(12.5, 6)

#histogram of the samples:
plt.subplot(211)
plt.title(r"Posterior distributions of the variables $\alpha, \beta$")
plt.hist(beta_samples, histtype='stepfilled', bins=35, alpha=0.85,
         label=r"posterior of $\beta$", color="#7A68A6", density=True)
plt.legend()

plt.subplot(212)
plt.hist(alpha_samples, histtype='stepfilled', bins=35, alpha=0.85,
         label=r"posterior of $\alpha$", color="#A60628", density=True)
plt.legend()


# t = np.linspace(temperature.min() - 5, temperature.max()+5, 50)[:, None]
# p_t = logistic(t.T, beta_samples, alpha_samples)
#
# mean_prob_t = p_t.mean(axis=0)
#
# plt.figure(2)
# # figsize(12.5, 4)
#
# plt.plot(t, mean_prob_t, lw=3, label="average posterior \nprobability \
# of defect")
# plt.plot(t, p_t[0, :], ls="--", label="realization from posterior")
# plt.plot(t, p_t[-2, :], ls="--", label="realization from posterior")
# plt.scatter(temperature, D, color="k", s=50, alpha=0.5)
# plt.title("Posterior expected value of probability of defect; \
# plus realizations")
# plt.legend(loc="lower left")
# plt.ylim(-0.1, 1.1)
# plt.xlim(t.min(), t.max())
# plt.ylabel("probability")
# plt.xlabel("temperature")
#
#
# # vectorized bottom and top 2.5% quantiles for "confidence interval"
# plt.figure(3)
# qs = mquantiles(p_t, [0.025, 0.975], axis=0)
# plt.fill_between(t[:, 0], *qs, alpha=0.7,
#                  color="#7A68A6")
#
# plt.plot(t[:, 0], qs[0], label="95% CI", color="#7A68A6", alpha=0.7)
#
# plt.plot(t, mean_prob_t, lw=1, ls="--", color="k",
#          label="average posterior \nprobability of defect")
#
# plt.xlim(t.min(), t.max())
# plt.ylim(-0.02, 1.02)
# plt.legend(loc="lower left")
# plt.scatter(temperature, D, color="k", s=50, alpha=0.5)
# plt.xlabel("temp, $t$")
#
# plt.ylabel("probability estimate")
# plt.title("Posterior probability estimates given temp. $t$")
plt.show()
