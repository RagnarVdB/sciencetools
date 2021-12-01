import numpy as np
import scipy.optimize as opt
from scipy.misc import derivative
from scipy import stats
from sciencetools.stats import chi2
from sciencetools.misc import *
from inspect import getfullargspec
import pandas as pd

pandasGlobal = True

def set_pandas(pandas):
    pandasGlobal = pandas

def _fitParameters(x, y, dy, model, guess, bounds=None, method=None, **kwargs):
    """minimaliseerd de chi2 functie"""
    minobj = opt.minimize(lambda p: chi2(x, y, dy, model, p), guess, **kwargs)
    return(minobj["x"])

def _errorFit2(x, y, dy, model, minimum):
    """Geeft onzekerheidsinterval rond fitparameters via tweede afgeleide"""
    uncertainties = []
    for i in range(len(minimum)):
        deriv = derivative(lambda p: chi2(x, y, dy, model, substitute(p, i, minimum)), minimum[i], n=2)
        uncertainties.append(np.sqrt(2/deriv))
    return np.array(uncertainties)

def _errorFit(x, y, dy, model, minimum, symmetric=False):
    """Geeft onzekerheidsinterval rond fitparameters via chi2 + 1"""
    uncertainties = []
    for i in range(len(minimum)):
        chi2_parameter = lambda p: chi2(x, y, dy, model, substitute(p, i, minimum))
        chi2_vector = np.vectorize(chi2_parameter)
        minval = (minimum[i], chi2_parameter(minimum[i]))
        f = lambda p: chi2_parameter(p) - minval[1] - 1
        left = opt.fsolve(f, minval[0] *0.9997)[0]
        right = opt.fsolve(f, minval[0]*1.0003)[0]
        if not symmetric:
            uncertainties.append(np.array([minval[0] - left, right - minval[0]]))
        else:
            avg = (right - left)/2
            uncertainties.append(avg)
    return np.array(uncertainties)

def fit(x, y, dy, model, guess=None, bounds=None, method=None, error_method=1, silent=False, pandas=True, **kwargs):
    """Voert fit uit voor gegeven dataset en model"""
    if type(dy) == int and dy == 0:
        dy = 1
        silent=True
    
    modelParams = getfullargspec(model).args
    if guess is None:
        guess = [1]*(len(modelParams) - 1)

    params = _fitParameters(x, y, dy, model, guess, bounds=bounds, method=method, **kwargs)
    if error_method == 1:
        errors = _errorFit(x, y, dy, model, params, symmetric=True)
    else:
        errors = _errorFit2(x, y, dy, model, params)

    if not silent:
        ls, rls, p = goodness_fit(x, y, dy, model, params)
        print("Least squares value: " + str(round(ls, 3)))
        print("Reduced Least squares: " + str(round(rls, 3)))
        print("p-value: " + str(round(p, 3)))
    
    arr = np.array([params, np.abs(errors)]).T
    if not pandas or not pandasGlobal:
        return arr
    else:
        return pd.DataFrame(arr, index=modelParams[1:], columns=["value", "error"])

def goodness_fit(x, y, dy, model, params):
    """Geeft chi2, reduced chi2 en p-waarde terug"""
    ls = chi2(x, y, dy, model, params)
    df = len(x) - len(params)
    p = 1 - stats.chi2.cdf(ls, df)
    return ls, ls/df, p