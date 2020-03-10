import pymc3 as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing

plt.style.use('seaborn-darkgrid')

nCPU = multiprocessing.cpu_count()
# # ----------------------------------------------------------------------------------
# # 4.1 Inferring a mean and standard deviation
# x = np.array([1.1, 1.9, 2.3, 1.8])
# n = len(x)
#
# with pm.Model() as model1:
#     # prior
#     mu = pm.Normal('mu', mu=0, tau=.001)
#     sigma = pm.Uniform('sigma', lower=0, upper=10)
#     # observed
#     xi = pm.Normal('xi', mu=mu, tau=1 / (sigma ** 2), observed=x)
#     # inference
#     trace = pm.sample(1000, cores=nCPU)
#
# pm.traceplot(trace[50:])
#
# y = trace['mu']
# x = trace['sigma']
# sns.jointplot(x, y, kind='scatter', stat_func=None, alpha=.5)
#
# print('The mu estimation is: %.3f' % y.mean())
# print('The sigma estimation is: %.3f' % x.mean())
#
# plt.show()
# # ----------------------------------------------------------------------------------


# # ----------------------------------------------------------------------------------
# # 4.2 The seven scientists
# # data
# x = np.array([-27.020, 3.570, 8.191, 9.898, 9.603, 9.945, 10.056])
# n = len(x)
#
# with pm.Model() as model2:
#     # prior
#     mu = pm.Normal('mu', mu=0, tau=.001)
#     # lambda1 = pm.Gamma('lambda1', alpha=.001, beta=.001, shape=n)
#     # sigma = pm.Deterministic('sigma',1 / np.sqrt(lambda1))
#
#     sigma = pm.Uniform('sigma', lower=0, upper=100)
#     lambda1 = 1 / (sigma ** 2)
#     # observed
#     xi = pm.Normal('xi', mu=mu, tau=lambda1, observed=x)
#
#     # inference
#     trace2 = pm.sample(1000, cores=nCPU)
#
# pm.traceplot(trace2)
# print(pm.summary(trace2))
#
#
# plt.show()
# # ----------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------
# 4.3 Repeated measurement of IQ
y = np.array([[90, 95, 100], [105, 110, 115], [150, 155, 160]])
ntest = 3
nsbj = 3

import sys

eps = sys.float_info.epsilon

with pm.Model() as model3:
    # mu_i ~ Uniform(0, 300)
    # notices the shape here need to be properly
    # initualized to have the right repeated measure
    # mui = pm.Uniform('mui', 0, 300, shape=(nsbj, 1))
    mui = pm.Normal('mui',mu=100, tau=.0044,shape=(nsbj,1))

    # sg ~ Uniform(0, 100)
    sg = pm.Uniform('sg', .0, 100)
    lambda1 = 1 / (sg ** 2)
    yd = pm.Normal('y', mu=mui, tau=lambda1, observed=y)

    # # It is more stable to use a Gamma prior
    # lambda1 = pm.Gamma('lambda1', alpha=.01, beta=.01)
    # sg = pm.Deterministic('sg', 1 / np.sqrt(lambda1))
    #
    # # y ~ Normal(mu_i, sg)
    # yd = pm.Normal('y', mu=mui, sd=sg, observed=y)

    trace3 = pm.sample(1000, cores=nCPU)

pm.traceplot(trace3)
print(pm.summary(trace3))
plt.show()
# ----------------------------------------------------------------------------------
