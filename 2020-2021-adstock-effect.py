#!/usr/bin/env python
# coding: utf-8

# _Using Python in Marketing Analysis | Marketing Data Science_
# 
# ***
# 
# ## Table of Contents
# 
# * [Introduction](#intro)
# * [Import Required Libraries](#import)
# * [Load Data](#load)
# * [Derive Adstock Rate with Analytical Method](#analytical)
# * [Plot](#plot)
# * [Prepare Data](#prep)
# * [Discover the Adstock Effect Using Neural Network](#nn)
# * [Solve with Probabilistic Programming](#prob)
# * [Estimate the Adstock and Hill Function Parameters](#params)
# 
# ## Introduction <a id='intro'></a>
# 
# An important question in marketing data science is how advertising affects sales. This problem often comes up as “I spend money on day 1, when will I see the sales/subscriptions go up, and by how much”. The money spend now may take time to have an effect. For example, it might take some time for ads to circulate in the community. The effect of advertising may also carry over from one ads campaign to the next because people remember your products. This kind of effect is called a lag/carryover effect.
# 
# Another kind of effect has to do with the diminishing returns on ads. That is to say, spending too much on ads will simply saturate the market and any more money spent won’t drive the sales further. This is called saturation effect.
# 
# ## Import Required Libraries <a id='import'></a>



import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(rc={'figure.figsize':(8, 3)})

import torch
import torch.nn as nn

import pyro
import pyro.distributions as dist
pyro.set_rng_seed(1)
pyro.enable_validation(True)
from pyro.infer import MCMC, NUTS
from pyro.nn import PyroModule, PyroSample


# Read the dataset and load it into a Pandas dataframe.



# Read specific columns
df = pd.read_csv('2020-2021_combined_data.csv', usecols=['Date', 'Ad Impressions', 'Direct Visit Clicks'])


# Set date as index
df = df.set_index(pd.DatetimeIndex(df['Date']))
df.drop('Date', axis=1, inplace=True)

# Normalize data into a 0-10 scale
df = 10 * (df - df.min()) / (df.max() - df.min())

# ## Derive Adstock Rate with Analytical Method <a id='analytical'></a>



# Create a function to calculate adstock values
def calc_adstock(ad_data, adstock_rate):
    # Init adstock
    adstock = np.zeros(len(ad_data))
    # Init first adstock value
    adstock[0] = ad_data[0]
    # Loop through the dataset to calculate adstock values
    for i in range(1, len(ad_data)):
        adstock[i] = ad_data[i] + adstock_rate * adstock[i-1]
    return adstock


# Create a function that optimizes adstock and makes a plot
def optimize_adstock(ad_data, clicks):
    # Init a list to keep correlations
    corrs = []
    # Create an array of rates to check for optimization
    rates = np.arange(0, 1.01, 0.01)
    # Init max rate
    rmax = 0
    # Init max correlation
    corrmax = -1
    # Loop through the rates
    for r in rates:
        # Calculate adstock for the specific rate
        x = calc_adstock(ad_data.copy(), r)
        # Get the clicks data
        y = clicks
        # Compute correlation
        corr = np.corrcoef(x, y)[0][1]
        # Append correlation to the list
        corrs.append(corr)
        # Find the maximum correlation
        if corr > corrmax:
            rmax = r
            corrmax = corr
    # Create the plot
    plt.title("Correlation vs. Decay Rate")
    plt.plot(rates, corrs, label="Correlations")
    plt.xlabel("Decay Rates")
    plt.ylabel("Correlations")
    plt.legend()
    plt.show()
    
    print('Found {0} decay rate at maximum correlation coefficient {1}'.format(rmax, corrmax))
    
    return rmax



# Optimize adstock for our dataset and print the decay rate
rmax = optimize_adstock(ad_data=df['Ad Impressions'], clicks=df['Direct Visit Clicks'])


# **Inference**
# 
# - Very low correlation


# Create the adstock data for the maximum decay rate
df['Ad Impressions After Adstock'] = calc_adstock(ad_data=df['Ad Impressions'], adstock_rate=rmax)


df.plot(secondary_y=['Ad Impressions', 'Ad Impressions After Adstock'], figsize=(14, 8));


 
# Another way to deal with it is to recover the adstock kernel from the ad and clicks data. This could be done using deep learning tools such as a neural network. First, we need to prepare/transform the data.



def get_padded_spend(spend, max_lag):
    """
    convert vector of spend/impressions to matrix where
    each column is one element with
    [spend[t], spend[t-1], …, spend[t-max_lag]]
    shape = (day x max_lag)
    """
    X_media = []
    N = len(spend)
    for time_point in range(N):
        unpadded = spend[max([0, time_point-max_lag]):(time_point + 1)]
        pad = [0]*max([0, max_lag + 1 - len(unpadded)])
        X_media.append(unpadded[::-1] + pad[::-1])
    return np.array(X_media)

# Pad ad data
matrix_spend = get_padded_spend(df['Ad Impressions'].to_list(), 5)
# Transform for convolutional neural network
torch_spend = torch.Tensor(df['Ad Impressions'].to_list())
# Transform for fully connected neural network
x_data = torch.Tensor(matrix_spend)
y_data = torch.Tensor(df['Direct Visit Clicks'].to_list())

# Here we first look at a single layer neural network to for the adstock function. I tried both linear or fully connected (fc) and convolutional (conv) neural network. In both cases, I used pytorch implementation of neural network. The code for this part is shown in the Appendix subsection D.
# 
# I trained the model for 5000 iterations. The results shown below are the weights of the neural network used to fit it. The weights should reflect the adstock kernel, which they indeed are.


get_ipython().run_cell_magic('time', '', "\nmodel = nn.Linear(5+1,1)\nmodel = model.to()\nlearning_rate = 1e-3\ncriterion = torch.nn.MSELoss()\noptimizer_fc = torch.optim.Adam(model.parameters(), lr=learning_rate)\n\nconv_model = nn.Conv1d(1, 1, 5+1, stride=1)\nlearning_rate = 1e-3\ncriterion = torch.nn.MSELoss()\noptimizer_conv = torch.optim.Adam(conv_model.parameters(), lr=learning_rate)\n\ndef execute_function(model,optimizer,epochs, data_X, data_y,conv=False):\n    epoch_loss = []\n    count = 0\n    if conv:\n        data_X = data_X.unsqueeze(0).unsqueeze(0)\n    for epoch in range(epochs):\n        temp_loss = []\n        model.train()\n        optimizer.zero_grad()\n        result = model(data_X)\n        #print(result.shape)\n        if conv:\n            result = result.squeeze(0).squeeze(0)\n        else:\n            result = result.squeeze(1)\n        loss = criterion(result,data_y)\n        loss.backward()\n        optimizer.step()\n        temp_loss.append(loss.item())\n        count += 1\n        #print(count)\n        epoch_loss.append(np.mean(temp_loss))\n    return epoch_loss\n\nloss1 = execute_function(conv_model,optimizer_conv,5000,torch_spend,y_data[5:],conv=True)\nloss2 = execute_function(model,optimizer_fc,5000,x_data,y_data)\n\n# to plot this we can get the weights of the models and plot the weights\n# note the conv. filter is the flipped kernel due to the notation convention of convolution.\n# we will just flip back\n# below here is the code plotting\n\nconv_filter = conv_model.weight.squeeze(0).squeeze(0).detach().numpy()[::-1]\nnp_kernel = model.weight.detach().numpy()\nnp_kernel = np_kernel.reshape(5+1)\nplt.plot(conv_filter,label = 'conv filter')\nplt.plot(np_kernel,label='fc filter')\n# plt.plot(kernel_carryover/np.sum(kernel_carryover),label = 'ground truth')\nplt.legend(loc='best')\nplt.title('single layer perception fitting')")


# In addition to the previous methods, we can use Bayesian inference. The added benefit of doing this is that we get the full probability distribution of the adstock kernels. It provides the ranges and uncertainty estimates of the models.

get_ipython().run_cell_magic('time', '', "\n# first we will specify the model as function\n# this is different from Pymc3 which uses with context manager\n\ndef bayesian_model_indexing(x_data,y_data,max_lag=5+1):\n    # the weights (shape = 11) are sampled independently from 11 normal distribution.\n    weight = pyro.sample('weight',dist.Normal(torch.zeros(max_lag),5.0*torch.ones(max_lag)))\n    with pyro.plate('data',len(x_data)):\n        mean = torch.sum(weight*x_data,dim=1)  # apply adstock kernel to the spend data (dot product)\n        pyro.sample('y',dist.Normal(mean,1),obs=y_data) # the result is the sales. subjected to observation\n\n\n# Then we will call create NUTS and MCMC objects which specify how we will run MCMC\nkernel_bayes= NUTS(bayesian_model_indexing)\nmcmc_bayes = MCMC(kernel_bayes, num_samples=1000, warmup_steps=200)\nmcmc_bayes.run(x_data, y_data) # don't forget to include your data in the run\n\n# we get the traces of MCMC and store as a dictionary \n# we then can use these traces to find means and credible intervals\nhmc_samples = {k: v.detach().cpu().numpy() for k, v in mcmc_bayes.get_samples().items()}\n\n# get the means value, and 90% credible interval\nmean_weight = np.mean(hmc_samples['weight'],axis=0)\nweight_0p1 = np.quantile(hmc_samples['weight'],q=0.1,axis=0)\nweight_0p9 = np.quantile(hmc_samples['weight'],q=0.9,axis=0)\n\n# finally we can plot this in Matplotlib\nfig, ax = plt.subplots()\nax.plot(mean_weight,label='mean weight')\nax.fill_between([i for i in range(5+1)],weight_0p1,weight_0p9, color='b', alpha=.1)\nax.legend(loc='best')\nplt.title('Bayesian estimation of the adstock function')\nplt.xlabel('days')\nplt.ylabel('weight')")


# Finally, the Bayesian inference could be done in hierarchical manners where the kernel weight is generated from hyper-priors, namely the retention rate, delay rate, etc. This allows us to estimate the parameters such as delay, retention, etc. that govern the spending/sales dynamics. With this approach we will also estimate the dimishing returns.



def bayesian_model_adstock(x_data,y_data,max_lag=5+1,fit_Hill=False):
    lag = torch.tensor(np.arange(0, max_lag + 1))
    # here instead of sampling weights, we will sample the adstock parameters
    # the parameters are sampled independently from normal distribution
    retain_rate = pyro.sample('retain_rate',dist.Uniform(0,1))
    delay = pyro.sample('delay',dist.Normal(1,5))

    # the adstock parameters are used to generate the adstock kernels
    weight = retain_rate**((lag-delay)**2)
    weight = weight/torch.sum(weight)

    # sample the saturation function parameters
    ec50 = pyro.sample('ec50',dist.Normal(0.5,1))
    slope = pyro.sample('Hill slope',dist.Normal(5,2.5))
    beta = pyro.sample('beta',dist.Normal(0,0.5))

    with pyro.plate('data',len(x_data)):
        # apply adstock kernel to spend data
        mean = torch.sum(weight*x_data,dim=1)
        # conditional, do you want to fit saturation function ?
        if fit_Hill:
            response = 1/(1+ (mean/ec50)**-slope)
            response = beta*response
        else:
            response = mean
        # the result is the sales which is subjected to the observation.
        pyro.sample('y', dist.Normal(response,1), obs=y_data)



get_ipython().run_cell_magic('time', '', '\nkernel_bayes= NUTS(bayesian_model_adstock)\nmcmc_bayes = MCMC(kernel_bayes, num_samples=1000, warmup_steps=200)\nmcmc_bayes.run(x_data, y_data, max_lag=5, fit_Hill=True)\n\nhmc_samples = {k: v.detach().cpu().numpy() for k, v in mcmc_bayes.get_samples().items()}')


for key, value in hmc_samples.items():
    print(key)
    sns.histplot(value, kde=True)
    plt.show()


# **Inference**
# 
# Look at the kde distributions for each parameter:
# 
# * The Adstock effect is carried over from time t to t+1 at the `retain_rate`
# * The peak effect can be delayed by the `delay` term
# * The half-saturation point is at `ec50`
# * The shape parameter is given by the `Hill slope`
