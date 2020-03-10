import pymc3 as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing

from matplotlib import gridspec

nCPU = multiprocessing.cpu_count()

# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'
plt.style.use('seaborn-darkgrid')

# # ----------------------------------------------------------------------------------
# # 3.1 Inferring a rate
# # Data
# k = np.array([0])
# n = np.array([1])
#
# with pm.Model() as model1:
#     # prior
#     theta = pm.Beta('theta', alpha=1, beta=1)
#     # observed
#     x = pm.Binomial('x', n=n, p=theta, observed=k)
#     # inference
#     trace1 = pm.sample(draws=500)
#
# pm.traceplot(trace1, var_names=['theta'])
#
# _, axes = plt.subplots(1, 2, figsize=(12, 3))
# pm.kdeplot(trace1['theta'], ax=axes[0])
# axes[0].hist(trace1['theta'], bins=100, density=1, alpha=.3)
# axes[0].set_xlabel('Rate')
# axes[0].set_ylabel('Posterior Density')
#
# pm.plot_posterior(trace1['theta'], ax=axes[1], color='#87ceeb')
# plt.tight_layout()
#
# print(pm.summary(trace1, var_names=['theta']).round(3))# gives the same credible interval as in the book.
# # ----------------------------------------------------------------------------------



# # ----------------------------------------------------------------------------------
# # 3.2 Difference between two rates
# # data
# k1, k2 = 5, 7 # number of successes
# n1 = n2 = 10 # total number of observations
#
# with pm.Model() as model2:
#     # prior
#     theta1 = pm.Beta('theta1', alpha=1, beta=1)
#     theta2 = pm.Beta('theta2', alpha=1, beta=1)
#     # observed
#     x1 = pm.Binomial('x1', n=n1, p=theta1, observed=k1)
#     x2 = pm.Binomial('x2', n=n2, p=theta2, observed=k2)
#     # differences as deterministic
#     delta = pm.Deterministic('delta', theta1-theta2)
#     # inference
#     trace2 = pm.sample(tune=1000)
#
# pm.traceplot(trace2)
# print(pm.summary(trace2).round(3))
#
# _, axes = plt.subplots(1, 2, figsize=(12, 3))
# pm.kdeplot(trace2['delta'], ax=axes[0])
# axes[0].hist(trace2['delta'], bins=100, density=1, alpha=.3)
# axes[0].set_xlabel('Difference in Rates')
# axes[0].set_ylabel('Posterior Density')
#
# pm.plot_posterior(trace2['delta'], ax=axes[1], color='#87ceeb')
# plt.tight_layout()
# plt.show()
# # ----------------------------------------------------------------------------------

# # ----------------------------------------------------------------------------------
# # 3.3 Inferring a common rate
# # Multiple trials
# k = np.array([0,10])
# n = np.array([10,10])
#
# with pm.Model() as model3:
#     # prior
#     theta = pm.Beta('theta', alpha=1, beta=1)
#     # observed
#     x = pm.Binomial('x', n=n, p=theta, observed=k)
#     # inference
#     trace3 = pm.sample(draws=1000)
#
# pm.traceplot(trace3, var_names=['theta'])
# print(pm.summary(trace3).round(3))
#
# _, axes = plt.subplots(1, 1, figsize=(6, 3))
# pm.kdeplot(trace3['theta'], ax=axes)
# axes.hist(trace3['theta'], bins=100, density=1, alpha=.3)
#
# axes.set_xlabel('Rate')
# axes.set_ylabel('Posterior Density')
#
# plt.show()
# # ----------------------------------------------------------------------------------

# # ----------------------------------------------------------------------------------
# # 3.4 Prior and posterior prediction
# k = 24
# n = 121
# # Uncomment for Trompetter Data
# # k = 24
# # n = 121
#
# # prior only model - no observation
# with pm.Model() as model_prior:
#     theta = pm.Beta('theta', alpha=1, beta=1)
#     x = pm.Binomial('x', n=n, p=theta)
#     trace_prior = pm.sample(draws=1000, compute_convergence_checks=False, tune=5000)
#
# # with observation
# with pm.Model() as model_obs:
#     theta = pm.Beta('theta', alpha=1, beta=1)
#     x = pm.Binomial('x', n=n, p=theta, observed=k)
#     trace_obs = pm.sample(draws=1000,tune=3000)
#
# # prediction (sample from trace)
# ppc = pm.sample_posterior_predictive(trace_obs, model=model_obs)
#
# print(pm.summary(ppc).round(3))
#
# from scipy.stats import beta
#
# prior_x = trace_prior['x']
# pred_theta = trace_obs['theta']
#
# _, axes = plt.subplots(2, 1, figsize=(6, 5))
#
# pm.kdeplot(pred_theta, ax=axes[0], label='Posterior')
# x = np.linspace(0, 1, 100)
# axes[0].plot(x, beta.pdf(x, 1, 1), color='b', label='Prior')
# axes[0].set_xlabel('Rate')
# axes[0].set_ylabel('Density')
#
# predictx = ppc['x']
# axes[1].hist(predictx, density=1, bins=len(np.unique(predictx)),
#          alpha=.3, color='r', label='Posterior')
# axes[1].hist(prior_x, density=1, bins=n+1,
#          alpha=.3, color='b', label='Prior')
# axes[1].set_xlabel('Success Count')
# axes[1].set_ylabel('Mass')
#
# plt.legend()
# plt.tight_layout()
# plt.show()
# # ----------------------------------------------------------------------------------

# # ----------------------------------------------------------------------------------
# # 3.5 Posterior predictive
# # Inferring a Common Rate, With Posterior Predictive
# k1 = 0
# n1 = 10
# k2 = 10
# n2 = 10
#
# with pm.Model() as model5:
#     # prior
#     theta = pm.Beta('theta', alpha=1, beta=1)
#     # observed
#     x1 = pm.Binomial('x1', n=n2, p=theta, observed=k1)
#     x2 = pm.Binomial('x2', n=n2, p=theta, observed=k2)
#     # inference
#     trace5 = pm.sample()
#
# pm.traceplot(trace5, varnames=['theta']);
# # prediction (sample from trace)
# ppc5 = pm.sample_ppc(trace5, samples=500, model=model5)
#
# fig = plt.figure(figsize=(12, 4))
# gs = gridspec.GridSpec(1,2, width_ratios=[2, 3])
# ax0 = plt.subplot(gs[0])
#
# pm.kdeplot(trace5['theta'], ax=ax0)
# ax0.hist(trace5['theta'], bins=100, normed=1, alpha=.3)
# plt.xlabel('Rate')
# plt.ylabel('Posterior Density')
#
# ax1 = plt.subplot(gs[1])
# predx1 = ppc5['x1']
# predx2 = ppc5['x2']
#
# from scipy import sparse
# A = sparse.csc_matrix((np.ones(len(predx1)), (predx1, predx2)),
#                       shape=(n1+1, n2+1)).todense()
# ax1.imshow(A, alpha=.9, origin='lower', cmap='viridis')
# ax1.scatter(k2, k1, s=100, c=[1,0,0])
# plt.xlabel('Trial2')
# plt.ylabel('Trial1')
# plt.tight_layout()
#
# plt.show()
# # ----------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------
# 3.6 Joing distributions
# the Survey example in the book
k = np.array([16, 18, 22, 25, 27]) # number of returned surveys
nmax = 500 # Maximum number of surveys
m = len(k)

with pm.Model() as model6:
    # prior
    theta = pm.Beta('theta', alpha=1, beta=1)
    TotalN = pm.DiscreteUniform('TotalN', lower=1, upper=nmax)
    # observed
    x = pm.Binomial('x', n=TotalN, p=theta, observed=k)
    # inference
    trace6 = pm.sample(draws=10000, cores=nCPU)

pm.traceplot(trace6)

burnin = 9500
thetapost = trace6['theta'][burnin:]
npost = trace6['TotalN'][burnin:]

g = sns.jointplot(npost, thetapost, kind='scatter', stat_func=None, alpha=.01)

from scipy.special import gammaln

cc = -float('Inf')
ind = 0

for i in range(0, len(npost)):
    logL = 0
    for j in k:
        logL = logL + gammaln(npost[i] + 1) - gammaln(j + 1) - gammaln(npost[i] - j + 1)
        logL = logL + j * np.log(thetapost[i]) + (npost[i] - j) * np.log(1 - thetapost[i])

    if logL > cc:
        ind = i
        cc = logL

g.ax_joint.plot(np.mean(npost), np.mean(thetapost), 'o', color='g') # expected values
g.ax_joint.plot(npost[ind], thetapost[ind], 'o', color='r') # maximum likelihood

plt.figure(1, figsize=(6, 6))
plt.hist2d(npost, thetapost, bins=50, cmap='viridis')

plt.show()
# ----------------------------------------------------------------------------------
