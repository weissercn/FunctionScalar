from FunctionScalar import *
import numpy as np
import matplotlib.pyplot as plt

mu, sigma  = 2., 0.5
n_points = 100
data = np.random.normal(mu, sigma, n_points)

#count, bins, ignored = plt.hist(s, 30, normed=True)
#plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) ), linewidth=2, color='r')
#plt.show()


#fs = FunctionScalar("unif")  # calling the uniform function by name
#fs = FunctionScalar("gauss") # calling the normal function by name 
#fs = FunctionScalar(TransformedFunction_Gauss()) # calling the normal function by class
fs = FunctionScalar(np.array(range(1,1+n_points))/(1.+n_points) ) # calling the uniform function directly by inv_cdf value


fs.fit(data)
data_transf = fs.transform(data)
data_invtransf = fs.invtransform(data_transf)

#count, bins, ignored = plt.hist(data, 30, normed=True)
#plt.hist(data_transf, 30, normed=True)
#count, bins, ignored = plt.hist(data_invtransf, 30, normed=True)
#plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) ), linewidth=2, color='r')


extrapolated_data = [2, -4, 6]
data_p_exp = np.append(extrapolated_data,data)
data_p_exp_transf = fs.transform(data_p_exp)
data_p_exp_invtransf = fs.invtransform(data_p_exp_transf)

#count, bins, ignored = plt.hist(data_p_exp, 30, normed=True)
#plt.hist(data_p_exp_transf, 30, normed=True)
count, bins, ignored = plt.hist(data_p_exp_invtransf, 30, normed=True)
#plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) ), linewidth=2, color='r')
plt.show()


