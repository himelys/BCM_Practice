import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pymc3.distributions.transforms as tr
import theano.tensor as tt
from scipy.special import gammaln

# rat data (BDA3, p. 102)
y = np.array([
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,
    1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  2,  1,  5,  2,
    5,  3,  2,  7,  7,  3,  3,  2,  9, 10,  4,  4,  4,  4,  4,  4,  4,
    10,  4,  4,  4,  5, 11, 12,  5,  5,  6,  5,  6,  6,  6,  6, 16, 15,
    15,  9,  4
])
n = np.array([
    20, 20, 20, 20, 20, 20, 20, 19, 19, 19, 19, 18, 18, 17, 20, 20, 20,
    20, 19, 19, 18, 18, 25, 24, 23, 20, 20, 20, 20, 20, 20, 10, 49, 19,
    46, 27, 17, 49, 47, 20, 20, 13, 48, 50, 20, 20, 20, 20, 20, 20, 20,
    48, 19, 19, 19, 22, 46, 49, 20, 20, 23, 19, 22, 20, 20, 20, 52, 46,
    47, 24, 14
])

N = len(n)

# Compute on log scale because products turn to sums
def log_likelihood(alpha, beta, y, n):
    LL = 0

    # Summing over data
    for Y, N in zip(y, n):
        LL += gammaln(alpha+beta) - gammaln(alpha) - gammaln(beta) + \
            gammaln(alpha+Y) + gammaln(beta+N-Y) - gammaln(alpha+beta+N)

    return LL


def log_prior(A, B):

    return -5/2*np.log(A+B)


def trans_to_beta(x, y):

    return np.exp(y)/(np.exp(x)+1)


def trans_to_alpha(x, y):

    return np.exp(x)*trans_to_beta(x, y)


# Create space for the parameterization in which we wish to plot
X, Z = np.meshgrid(np.arange(-2.3, -1.3, 0.01), np.arange(1, 5, 0.01))
param_space = np.c_[X.ravel(), Z.ravel()]
df = pd.DataFrame(param_space, columns=['X', 'Z'])

# Transform the space back to alpha beta to compute the log-posterior
df['alpha'] = trans_to_alpha(df.X, df.Z)
df['beta'] = trans_to_beta(df.X, df.Z)

df['log_posterior'] = log_prior(
    df.alpha, df.beta) + log_likelihood(df.alpha, df.beta, y, n)
df['log_jacobian'] = np.log(df.alpha) + np.log(df.beta)

df['transformed'] = df.log_posterior+df.log_jacobian
df['exp_trans'] = np.exp(df.transformed - df.transformed.max())

# This will ensure the density is normalized
df['normed_exp_trans'] = df.exp_trans/df.exp_trans.sum()


surface = df.set_index(['X', 'Z']).exp_trans.unstack().values.T

fig, ax = plt.subplots(figsize=(8, 8))
ax.contourf(X, Z, surface)
ax.set_xlabel(r'$\log(\alpha/\beta)$', fontsize=16)
ax.set_ylabel(r'$\log(\alpha+\beta)$', fontsize=16)

ix_z, ix_x = np.unravel_index(np.argmax(surface, axis=None), surface.shape)
ax.scatter([X[0, ix_x]], [Z[ix_z, 0]], color='red')

text = r"$({a},{b})$".format(a=np.round(
    X[0, ix_x], 2), b=np.round(Z[ix_z, 0], 2))

ax.annotate(text,
            xy=(X[0, ix_x], Z[ix_z, 0]),
            xytext=(-1.6, 3.5),
            ha='center',
            fontsize=16,
            color='black',
            arrowprops={'facecolor':'white'}
            )


#Estimated mean of alpha
print((df.alpha*df.normed_exp_trans).sum().round(3))

#Estimated mean of beta
print((df.beta*df.normed_exp_trans).sum().round(3))

def logp_ab(value):
    ''' prior density'''
    return tt.log(tt.pow(tt.sum(value), -5/2))


with pm.Model() as model:
    # Uninformative prior for alpha and beta
    ab = pm.HalfFlat('ab',
                     shape=2,
                     testval=np.asarray([1., 1.]))
    pm.Potential('p(a, b)', logp_ab(ab))

    X = pm.Deterministic('X', tt.log(ab[0]/ab[1]))
    Z = pm.Deterministic('Z', tt.log(tt.sum(ab)))

    alpha = pm.Deterministic('alpha', ab[0])
    beta = pm.Deterministic('beta', ab[1])

    theta = pm.Beta('theta', alpha=ab[0], beta=ab[1], shape=N)

    p = pm.Binomial('y', p=theta, observed=y, n=n)
    trace = pm.sample(1000, tune=2000, target_accept=0.95)

# Check the trace. Looks good!
# pm.traceplot(trace, var_names=['ab', 'X', 'Z'])
pm.traceplot(trace, var_names=['alpha','beta'])

print(pm.summary(trace, var_names=['alpha', 'beta']))

plt.figure()
sns.kdeplot(trace['X'], trace['Z'], shade=True, cmap='viridis')
# begi = 4000
#
# alpha_trunc = trace['beta'][begi: begi + 1000]
# beta_trunc = trace['alpha'][begi: begi + 1000]
#
# fig, ax = plt.subplots(figsize=(8, 7))
# sns.kdeplot(np.log(alpha_trunc / beta_trunc), np.log(alpha_trunc + beta_trunc),
#             cmap=plt.cm.viridis, n_levels=11, ax=ax)
# ax.set_xlim(1.5, 2.1)
# ax.set_ylim(1.7, 3.75)
# ax.set_xlabel('log(alpha / beta)')
# ax.set_ylabel('log(alpha + beta)')

pm.plot_posterior(trace, var_names=['alpha','beta'])

# estimate the means from the samples
print(trace['ab'].mean(axis=0))

post_theta = pm.sample_ppc(trace, samples=1000, model=model, vars=[theta], progressbar=False)
median_theta = []

for i in range(post_theta['theta'].shape[1]):
    median_theta.append(np.median(post_theta['theta'][:, i]))

error_up = []
error_down = []

for i in range(post_theta['theta'].shape[1]):
    a, b = pm.hpd(post_theta['theta'][:, i])
    error_up.append(b)
    error_down.append(a)

plt.figure(figsize=(10, 8))
plt.errorbar(y / n, median_theta, fmt='o',
             yerr=[error_down, error_up], ecolor='C5', markerfacecolor='k', mec='k',
             errorevery=1)
plt.plot(np.linspace(0, .4, 10), np.linspace(0, .4, 10),
         color='k')
plt.xlabel('Observed rate, y(i) / N(i)', fontsize=14)
plt.ylabel('95% hightest posterior density for each theta', fontsize=14);

plt.show()