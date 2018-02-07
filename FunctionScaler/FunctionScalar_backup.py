from scipy.interpolate import interp1d
from scipy.stats import norm
import numpy as np

class TransformedFunction:
    def __init__(self):
        pass
    def inv_cdf(self,n_points=100):
        print "This should be implemented in the daughter class"
        return None


class TransformedFunction_Uniform(TransformedFunction):
        
    def inv_cdf(self, n_points=100):
        inv_cdf_points = np.array(range(1,1+n_points))/(1.+n_points)
        return inv_cdf_points


class TransformedFunction_Gauss(TransformedFunction):
        
    def inv_cdf(self, n_points=100):
        # ppf is the inverse of cdf as shown by
        # norm.cdf(norm.ppf(0.95)) = 0.95
        # scipy.stats.norm.ppf = scipy.special.ndtri
        inv_cdf_points = norm.ppf( np.array(range(1,1+n_points))/(1.+n_points)  )
        return inv_cdf_points
        
def name_to_TransformedFunction(name):
    if name=="gauss" or name=="normal": return TransformedFunction_Gauss()
    elif name=="unif" or name=="uniform": return TransformedFunction_Uniform()

class FunctionScalar:
    def __init__(self, aTransformedFunction):
        if isinstance(aTransformedFunction, TransformedFunction):
            self.aTransformedFunction = aTransformedFunction
        elif isinstance(aTransformedFunction, str):
            self.aTransformedFunction = name_to_TransformedFunction(aTransformedFunction)
        else:
            self.aTransformedFunction = aTransformedFunction
        

    def fit(self, data):
        data = np.array(data)
        n_points = data.shape[0]
        #n_dims   = data.shape[1]
        #np.atleast_2d(np.array([1,2])).shape       
        self.LearnedFunctions = [] 
        self.LearnedInvFunctions = []


        data = np.sort(data)
        if isinstance(self.aTransformedFunction, TransformedFunction):
            y = self.aTransformedFunction.inv_cdf(n_points)
        else:
            y = np.array(self.aTransformedFunction)     

        #for d in n_dims:

        self.LearnedFunctions.append(interp1d( data , y, kind='linear', fill_value="extrapolate"))
        self.LearnedInvFunctions.append(interp1d( y, data, kind='linear', fill_value="extrapolate"))


    def transform(self, data):
        return self.LearnedFunctions[0](data)

    def fittransform(self, data):
        fit(data)
        return transform(data)

    def invtransform(self, data):
        return self.LearnedInvFunctions[0](data)


