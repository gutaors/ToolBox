#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# ## Caderno Preparing for statistical interview la do datacamp
# 
# probabilidade e distribuição de amostras
# análise de dados exploratória
# experimentos estatísticos
# regressão e classificação
# 
# 
# 

# ## Probability and Sampling Distributions

# ### Bayes' theorem applied
# ### é aquele do p de a dado b - p(a|b) = p(b|a)p(a)    /     p(b)
# 
# 
# You have two coins in your hand. Out of the two coins, one is a real coin and the second one is a faulty one with tails on both sides.
# 
# You are blindfolded and forced to choose a random coin and then toss it in the air. The coin falls down with tails facing upwards. Our goal is to find the probability that this is the faulty coin.

# In[1]:


# Print P(tails)
print(3/4)

# Print P(faulty)
print(1/2)

# Print P(tails | faulty)
print(1)

# Print P(faulty | tails)
print(2/3)


# In[2]:


#Use Bayes' theorem to solve this problem.
#Write out all the separate pieces before plugging into the formula.
#Make sure you have the formula correct and memorized for future use.


# ### Samples from a rolled die
# 
# Generate a sample of 10 die rolls and assign it to our small variable.
# 
# Assign the mean of the sample to a variable named small_mean and print the results; notice how close it is to the true mean.
# 
# Similarly, create a larger sample of 1000 die rolls and assign the list to a variable named large.
# 
# Assign the mean of the larger sample to a variable named large_mean and print the mean; which theorem is at work here?

# In[5]:


from numpy.random import randint

# Create a sample of 10 die rolls
small = randint(1, 7, size = 10)

# Calculate and print the mean of the sample
print(small.mean())


# Create a sample of 1000 die rolls
large = randint(1, 7, size = 1000)

# Calculate and print the mean of the large sample
print(large.mean())


#  ### the mean of the large sample has gotten closer to the true expected mean value of 3.5 for a rolled die. Which theorem did you say was being demonstrated here? Was it the law of large numbers? If so, you're correct! It's important to distinguish between the law of large numbers and central limit theorem in interviews.

# ### Simulating central limit theorem

# Se temos uma coleção grande o suficiente de amostras da mesma população, a distribuição das médias vai ser normalmente distribuida.
# Baseado nisto podemos fazer teste de hipóteses e rejeitar ou não nossas hipóiteses se ela vier ou não de uma distribuição particular. Isto é usado para teste AB.
# É diferente da teoria dos grandes números que diz que quanto mais amostras, mais a média das amostras reflete a média da população. 
# 
# Explicação da Khan Academy
# O teorema nos diz que se temos qualquer função que tenha uma VARIÂNCIA e uma MÉDIA bem definidas, logo um DESVIO PADRÃO bem definido, o gráfico de frequencia amostral tende a uma distribuição normal
# pode ser uma distribuição contínua ou discreta
# Então vamos pegar a Função de distribuição de probabilidades,
# a média mi está no meio
# vamos tomar algumas amostras e fazer a média das amostras e a frequencia que elas aparecem. O gráfico que vamos plotar é destas frequencias
# se somarmos várias ações em conjunto, assumindo que todas elas têm a mesma distribuição,  o gráfico vai ser uma distribuição normal, principalmente quando a distribuição tende a infinito.
# Fiz o exemplo para média amostral mas também vale para soma amostral ou qualquer distribuição.
# 
# Create a list named means with 1000 sample means from samples of size 30 by using list comprehension.

# In[6]:


from numpy.random import randint

# Create a list of 1000 sample means of size 30
means = [randint(1, 7, 30).mean() for i in range(1000)]


# Create and show a matplotlib histogram of the means; examine the shape of the distribution.

# In[7]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.hist(means)
plt.show()


# Adapt your code to visualize only 100 samples; did the distribution change at all?

# In[8]:


from numpy.random import randint

# Adapt code for 100 samples of size 30
means = [randint(1, 7, 30).mean() for i in range(100)]

# Create and show a histogram of the means
plt.hist(means)
plt.show()


# ### Bernoulli distribution
# Probabilidade discreta que descreve a probabilidade com duas possibilidades (cara e coroa por ex)
# A probabilidade de uma (cara) é igual a um menos a probabilidade da outra (coroa)
# cara = 1-coroa
# 
# Let's start simple with the Bernoulli distribution. In this exercise, you'll generate sample data for a Bernoulli event and then examine the visualization produced. Before we start, make yourself familiar with the rvs() function within scipy.stats that we'll use for sampling over the next few exercises.
# 
# Let's stick to the prior example of flipping a fair coin and checking the outcome: heads or tails.
# 
# 

# Generate a sample using the rvs() function with size set to 100; assign it to the data variable.

# In[9]:


# Generate bernoulli data
from scipy.stats import bernoulli
data = bernoulli.rvs(p=0.5, size=100)


# Create and display a matplotlib histogram; examine the shape of the distribution.

# In[10]:


# Plot distribution
plt.hist(data)
plt.show()


# Adapt the code to take a sample of 1000 observations this time.

# In[3]:


# Generate bernoulli data
from scipy.stats import bernoulli
data = bernoulli.rvs(p=0.5, size=1000)

# Plot distribution
plt.hist(data)
plt.show()


# ### distribioção Binomial
# ### Binomial distribution
# 
# é a soma de muitas amostras de bernoulli
#  

# In[6]:


from scipy.stats import binom 

#os três argumentos são respectivamente:
#-numero de sucessos desejado, é o “x” na fórmula;
#-numero de realizações do experimento, ou seja, o “n” da fórmula
#-probabilidade de sucesso em uma tentativa, o “p” na fórmula.

#se jogar uma moeda 5 vezes, qual a chance de dar cara 3 vezes?
binom.pmf(3,5,0.5)
dbinom(3,5,0.5)


# In[12]:


# Generate binomial data
from scipy.stats import binom
data = binom.rvs(n=10, p=0.8, size=1000)

# Plot the distribution
plt.hist(data)
plt.show()

# Assign and print probability of 8 or less successes
prob1 = binom.cdf(k=8, n=10, p=0.8)
print(prob1)

# Assign and print probability of all 10 successes
prob2 = binom.pmf(k=10, n=10, p=0.8)
print(prob2)


# ### Normal distribution
# é uma curva de sino com uma distribuição contínua. representa amostragem e teste de hipóteses.
# 68% das observações estão a 1 desvio padrão da média ( 6 8, dois maiores algarismos pares)
# 95 estão a 2 desvios padrão da média
# 99,7 estão a 3 desvios padrão da média

# In[13]:


# Generate normal data
from scipy.stats import norm
data = norm.rvs(size=1000)

# Plot distribution
plt.hist(data)
plt.show()

# Compute and print true probability for greater than 2
true_prob = 1 - norm.cdf(2)
print(true_prob)

# Compute and sample probability for greater than 2
sample_prob = sum(obs > 2 for obs in data) / len(data)
print(sample_prob)


# ## Exploratory Data Analysis

# ### Mean or median

# In[14]:


import pandas as pd
weather = pd.read_csv('dados/weatherAUS.csv')
weather.head()


# In[15]:


# Visualize the distribution 
plt.hist(weather['Temp3pm'])
plt.show()

# Assign the mean to the variable and print the result
print(weather['Temp3pm'].mean())


# Assign the median to the variable and print the result
print(weather['Temp3pm'].median())


# ### Standard deviation by hand

# In[16]:


# Create a sample list
import math
import numpy as np
nums = [1, 2, 3, 4, 5]

# Compute the mean of the list
mean = sum(nums) / len(nums)

# Compute the variance and print the std of the list
variance = sum(pow(x - mean, 2) for x in nums) / len(nums)
std = math.sqrt(variance)
print(std)

# Compute and print the actual result from numpy
real_std = np.array(nums).std()
print(real_std)


# ### Encoding techniques

# In[17]:


import pandas as pd
laptops = pd.read_csv('dados/laptops.csv', encoding = "ISO-8859-1", index_col = 0) #[['Company', 'Product', 'Price']]
laptops.head()


# In[18]:


from sklearn import preprocessing

# Create the encoder and print our encoded new_vals
encoder = preprocessing.LabelEncoder()
new_vals = encoder.fit_transform(laptops['Company'])
print(new_vals)


# In[19]:


# One-hot encode Company for laptops2
laptops2 = pd.get_dummies(data=laptops, columns=['Company'])
laptops2.head()


# ### Exploring laptop prices

# In[20]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
# Get some initial info about the data
laptops.info()

# Produce a countplot of companies
sns.countplot(laptops['Company'])
plt.show()


# In[21]:


# Visualize the relationship with price
laptops.boxplot('Price_euros', 'Company', rot=30)
plt.show()


# ### Types of relationships

# In[22]:


# Display a scatter plot and examine the relationship
plt.scatter(weather['MinTemp'], weather['Humidity3pm'])
plt.show()


# ### Pearson correlation

# In[23]:


# Generate the pair plot for the weather dataset
# sns.pairplot(weather)
# plt.show()


# In[24]:


# Compute and print the Pearson correlation
r = weather['Humidity9am'].corr(weather['Humidity3pm'])

# Calculate the r-squared value and print the result
r2 = r ** 2
print(r2)


# ### Sensitivity to outliers

# In[25]:


# Drop the outlier from the dataset
df = df.drop(2)

# Compute and print the correlation once more
new_corr  = df['X'].corr(df['Y'])
print(new_corr)


# ## Statistical Experiments and Significance Testing

# ### Confidence interval by hand

# In[ ]:


from scipy.stats import sem, t
data = [1, 2, 3, 4, 5]
confidence = 0.95

z_score = 2.7764451051977987
sample_mean = np.mean(data)

# Compute the standard error and margin of error
std_err = sem(data)
margin_error = std_err * z_score

# Compute and print the lower threshold
lower = sample_mean - margin_error
print(lower)

# Compute and print the upper threshold
upper = sample_mean + margin_error
print(upper)


# ### Applying confidence intervals

# In[ ]:


def proportion_confint(count, nobs, alpha=0.05, method='normal'):
    '''confidence interval for a binomial proportion

    Parameters
    ----------
    count : int or array_array_like
        number of successes, can be pandas Series or DataFrame
    nobs : int
        total number of trials
    alpha : float in (0, 1)
        significance level, default 0.05
    method : string in ['normal']
        method to use for confidence interval,
        currently available methods :

         - `normal` : asymptotic normal approximation
         - `agresti_coull` : Agresti-Coull interval
         - `beta` : Clopper-Pearson interval based on Beta distribution
         - `wilson` : Wilson Score interval
         - `jeffreys` : Jeffreys Bayesian Interval
         - `binom_test` : experimental, inversion of binom_test

    Returns
    -------
    ci_low, ci_upp : float, ndarray, or pandas Series or DataFrame
        lower and upper confidence level with coverage (approximately) 1-alpha.
        When a pandas object is returned, then the index is taken from the
        `count`.

    Notes
    -----
    Beta, the Clopper-Pearson exact interval has coverage at least 1-alpha,
    but is in general conservative. Most of the other methods have average
    coverage equal to 1-alpha, but will have smaller coverage in some cases.

    The 'beta' and 'jeffreys' interval are central, they use alpha/2 in each
    tail, and alpha is not adjusted at the boundaries. In the extreme case
    when `count` is zero or equal to `nobs`, then the coverage will be only
    1 - alpha/2 in the case of 'beta'.

    The confidence intervals are clipped to be in the [0, 1] interval in the
    case of 'normal' and 'agresti_coull'.

    Method "binom_test" directly inverts the binomial test in scipy.stats.
    which has discrete steps.

    TODO: binom_test intervals raise an exception in small samples if one
       interval bound is close to zero or one.

    References
    ----------
    http://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval

    Brown, Lawrence D.; Cai, T. Tony; DasGupta, Anirban (2001). "Interval
        Estimation for a Binomial Proportion",
        Statistical Science 16 (2): 101–133. doi:10.1214/ss/1009213286.
        TODO: Is this the correct one ?

    '''

    pd_index = getattr(count, 'index', None)
    if pd_index is not None and hasattr(pd_index, '__call__'):
        # this rules out lists, lists have an index method
        pd_index = None
    count = np.asarray(count)
    nobs = np.asarray(nobs)

    q_ = count * 1. / nobs
    alpha_2 = 0.5 * alpha

    if method == 'normal':
        std_ = np.sqrt(q_ * (1 - q_) / nobs)
        dist = stats.norm.isf(alpha / 2.) * std_
        ci_low = q_ - dist
        ci_upp = q_ + dist

    elif method == 'binom_test':
        # inverting the binomial test
        def func(qi):
            return stats.binom_test(q_ * nobs, nobs, p=qi) - alpha
        if count == 0:
            ci_low = 0
        else:
            ci_low = optimize.brentq(func, float_info.min, q_)
        if count == nobs:
            ci_upp = 1
        else:
            ci_upp = optimize.brentq(func, q_, 1. - float_info.epsilon)

    elif method == 'beta':
        ci_low = stats.beta.ppf(alpha_2, count, nobs - count + 1)
        ci_upp = stats.beta.isf(alpha_2, count + 1, nobs - count)

        if np.ndim(ci_low) > 0:
            ci_low[q_ == 0] = 0
            ci_upp[q_ == 1] = 1
        else:
            ci_low = ci_low if (q_ != 0) else 0
            ci_upp = ci_upp if (q_ != 1) else 1

    elif method == 'agresti_coull':
        crit = stats.norm.isf(alpha / 2.)
        nobs_c = nobs + crit**2
        q_c = (count + crit**2 / 2.) / nobs_c
        std_c = np.sqrt(q_c * (1. - q_c) / nobs_c)
        dist = crit * std_c
        ci_low = q_c - dist
        ci_upp = q_c + dist

    elif method == 'wilson':
        crit = stats.norm.isf(alpha / 2.)
        crit2 = crit**2
        denom = 1 + crit2 / nobs
        center = (q_ + crit2 / (2 * nobs)) / denom
        dist = crit * np.sqrt(q_ * (1. - q_) / nobs + crit2 / (4. * nobs**2))
        dist /= denom
        ci_low = center - dist
        ci_upp = center + dist

    # method adjusted to be more forgiving of misspellings or incorrect option name
    elif method[:4] == 'jeff':
        ci_low, ci_upp = stats.beta.interval(1 - alpha, count + 0.5,
                                             nobs - count + 0.5)

    else:
        raise NotImplementedError('method "%s" is not available' % method)

    if method in ['normal', 'agresti_coull']:
        ci_low = np.clip(ci_low, 0, 1)
        ci_upp = np.clip(ci_upp, 0, 1)
    if pd_index is not None and np.ndim(ci_low) > 0:
        import pandas as pd
        if np.ndim(ci_low) == 1:
            ci_low = pd.Series(ci_low, index=pd_index)
            ci_upp = pd.Series(ci_upp, index=pd_index)
        if np.ndim(ci_low) == 2:
            ci_low = pd.DataFrame(ci_low, index=pd_index)
            ci_upp = pd.DataFrame(ci_upp, index=pd_index)

    return ci_low, ci_upp


# In[ ]:


from scipy.stats import binom
import numpy as np
from scipy import stats
# Repeat this process 10 times 
heads = binom.rvs(50, 0.5, size=10)
for val in heads:
    confidence_interval = proportion_confint(val, 50, .10)
    print(confidence_interval)


# ### One tailed z-test

# In[ ]:


import pandas as pd
results = pd.read_csv('dados/ab_data.csv')[['group', 'converted']]
results.columns = ['Group', 'Converted']


# In[ ]:


# Assign and print the conversion rate for each group
conv_rates = results.groupby('Group').Converted.mean()


# In[ ]:


# Assign the number of conversions and total trials
num_control = results[results['Group'] == 'control']['Converted'].sum()
total_control = len(results[results['Group'] == 'control'])

# Assign the number of conversions and total trials
num_treat = results[results['Group'] == 'treatment']['Converted'].sum()
total_treat = len(results[results['Group'] == 'treatment'])

from statsmodels.stats.proportion import proportions_ztest
count = np.array([num_treat, num_control]) 
nobs = np.array([total_treat, total_control])

# Run the z-test and print the result 
stat, pval = proportions_ztest(count, nobs)
print('{0:0.3f}'.format(pval))


# ### Two tailed t-test

# In[ ]:


# Display the mean price for each group
prices = laptops.groupby('Company').Price_euros.mean()
print(prices)


# ### Two tailed t-test

# In[ ]:


# Assign the prices of each group
asus = laptops[laptops['Company'] == 'Asus']['Price_euros']
toshiba = laptops[laptops['Company'] == 'Toshiba']['Price_euros']

# Run the t-test
from scipy.stats import ttest_ind
tstat, pval = ttest_ind(asus, toshiba)
print('{0:0.3f}'.format(pval))


# ### Calculating sample size

# In[ ]:


# Standardize the effect size
from statsmodels.stats.proportion import proportion_effectsize
std_effect = proportion_effectsize(.20, .25)

# Assign and print the needed sample size
from statsmodels.stats.power import  zt_ind_solve_power
sample_size = zt_ind_solve_power(effect_size=std_effect, nobs1=None, alpha=.05, power=.95)
print(sample_size)


# In[ ]:


# Standardize the effect size
from statsmodels.stats.proportion import proportion_effectsize
std_effect = proportion_effectsize(.20, .25)

# Assign and print the needed sample size
from statsmodels.stats.power import  zt_ind_solve_power
sample_size = zt_ind_solve_power(effect_size=std_effect, nobs1=None, alpha=.05, power=0.8)
print(sample_size)


# ### Visualizing the relationship

# In[ ]:


sample_sizes = np.array(range(5, 100))
effect_sizes = np.array([0.2, 0.5, 0.8])

# Create results object for t-test analysis
from statsmodels.stats.power import TTestIndPower
results = TTestIndPower(nobs = sample_sizes, effect_size = effect_sizes)

# Plot the power analysis
results.plot_power(dep_var='nobs', nobs=sample_sizes, effect_size=effect_sizes)


# ### Calculating error rates

# Compute and print the probability of your colleague getting a Type I error.

# In[ ]:


# Print error rate for 60 tests with 5% significance
error_rate = 1 - (.95**(60))
print(error_rate)


# ### Bonferroni correction

# In[ ]:


from statsmodels.sandbox.stats.multicomp import multipletests
pvals = [.01, .05, .10, .50, .99]

# Create a list of the adjusted p-values
p_adjusted = multipletests(pvals, alpha=.05, method='bonferroni')

# Print the resulting conclusions
print(p_adjusted[0])

# Print the adjusted p-values themselves 
print(p_adjusted[1])


# ## Regression and Classification
# 

# ### Linear regression

# In[ ]:


from sklearn.linear_model import LinearRegression 
X_train = np.array(weather['Humidity9am'].dropna()[:1000]).reshape(-1,1)
y_train = weather['Humidity3pm'].dropna()[:1000]

# Create and fit your linear regression model
lm = LinearRegression()
lm.fit(X_train, y_train)

# Assign and print predictions
preds = lm.predict(X_train)

# Assign and print coefficient 
coef = lm.coef_
print(coef)


# In[ ]:


# Plot your fit to visualize your model
plt.scatter(X_train, y_train)
plt.plot(X_train, preds, color='red')
plt.show()


# ### Logistic regression

# In[ ]:


from sklearn.linear_model import LogisticRegression

# Create and fit your model
clf = LogisticRegression(solver='lbfgs')
clf.fit(X_train, y_train.values.ravel())

coefs = clf.coef_

# Compute and print the accuracy
# acc = clf.score(X_test, y_test)
# print(acc)


# ### Regression evaluation

# In[ ]:


# R-squared score
r2 = lm.score(X_train, y_train)
print(r2)


# In[ ]:


# Mean squared error
from sklearn.metrics import mean_squared_error
preds = lm.predict(X_train)
mse = mean_squared_error(y_train, preds)
print(mse)


# In[ ]:


# Mean absolute error
from sklearn.metrics import mean_absolute_error
preds = lm.predict(X_train)
mae = mean_absolute_error(y_train, preds)
print(mae)


# ### Classification evaluation

# In[ ]:


# Generate and output the confusion matrix
from sklearn.metrics import confusion_matrix
preds = clf.predict(X_test)
matrix = confusion_matrix(y_test, preds)
print(matrix)


# In[ ]:


# Compute and print the precision
from sklearn.metrics import precision_score
preds = clf.predict(X_test)
precision = precision_score(y_test, preds)
print(precision)


# In[ ]:


# Compute and print the recall
from sklearn.metrics import recall_score
preds = clf.predict(X_test)
recall = recall_score(y_test, preds)
print(recall)


# ### Handling null values

# In[ ]:


# Identify and print the the rows with null values
nulls = laptops[laptops.isnull().any(axis=1)]
print(nulls)


# In[ ]:


# Impute constant value 0
laptops.fillna(0, inplace=True)


# In[ ]:


# Impute median price
laptops.fillna(laptops.median(), inplace=True)


# In[ ]:


# Drop each row with a null value
laptops.dropna(inplace=True)


# ### Identifying outliers

# In[ ]:


laptops = laptops.rename(columns = {'Price_euros': 'Price'})
laptops.head()


# In[ ]:


laptops.rename({'Price_euros': 'Price'}, inplace = True)
# Calculate the mean and std
mean, std = laptops['Price'].mean(), laptops['Price'].std()

# Compute and print the upper and lower threshold
cut_off = std * 3
lower, upper = mean - cut_off, mean + cut_off
print(lower, 'to', upper)

# Identify and print rows with outliers
outliers = laptops[(laptops['Price'] > upper) | 
                   (laptops['Price'] < lower)]
print(outliers)

# Drop the rows from the dataset
laptops = laptops[(laptops['Price'] <= upper) | 
                  (laptops['Price'] >= lower)]


# ### Visualizing the tradeoff

# In[ ]:


# Use X and y to create a scatterplot
plt.scatter(X, y)

# Add your model predictions to the scatter plot 
plt.plot(np.sort(X), preds)

# Add the higher-complexity model predictions as well
plt.plot(np.sort(X), preds2)
plt.show()

