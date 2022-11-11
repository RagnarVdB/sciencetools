import numpy as np
import scipy.optimize as opt
from scipy.misc import derivative
from scipy import stats
from sciencetools.stats import chi2
from sciencetools.misc import substitute
from inspect import getfullargspec
import pandas as pd

pandasGlobal = True


def set_pandas(pandas):
    pandasGlobal = pandas


def gradient_respecting_bounds(bounds, fun, eps=1e-8):
    """bounds: list of tuples (lower, upper)"""

    def gradient(x):
        fx = fun(x)
        grad = np.zeros(len(x))
        for k in range(len(x)):
            d = np.zeros(len(x))
            d[k] = eps if x[k] + eps <= bounds[k][1] else -eps
            grad[k] = (fun(x + d) - fx) / d[k]
        return grad

    return gradient


def _fitParameters(
    x, y, dy, model, guess, bounds=None, method=None, respect_bounds=False, **kwargs
):
    """minimaliseerd de chi2 functie"""
    f = lambda p: chi2(x, y, dy, model, p)
    if respect_bounds:
        jac = gradient_respecting_bounds(bounds, f)
        minobj = opt.minimize(f, guess, bounds=bounds, jac=jac, **kwargs)
    else:
        minobj = opt.minimize(f, guess, bounds=bounds, **kwargs)
    return minobj


def _errorFit2(x, y, dy, model, minimum):
    """Geeft onzekerheidsinterval rond fitparameters via tweede afgeleide"""
    uncertainties = []
    for i in range(len(minimum)):
        deriv = derivative(
            lambda p: chi2(x, y, dy, model, substitute(p, i, minimum)), minimum[i], n=2
        )
        uncertainties.append(np.sqrt(2 / deriv))
    return np.array(uncertainties)


def _errorFit(x, y, dy, model, minimum, symmetric=False):
    """Geeft onzekerheidsinterval rond fitparameters via chi2 + 1"""
    uncertainties = []
    for i in range(len(minimum)):
        chi2_parameter = lambda p: chi2(x, y, dy, model, substitute(p, i, minimum))
        minval = (minimum[i], chi2_parameter(minimum[i]))
        f = lambda p: chi2_parameter(p) - minval[1] - 1
        left = opt.fsolve(f, minval[0] * 0.9997)[0]
        right = opt.fsolve(f, minval[0] * 1.0003)[0]
        if not symmetric:
            uncertainties.append(np.array([minval[0] - left, right - minval[0]]))
        else:
            avg = (right - left) / 2
            uncertainties.append(avg)
    return np.array(uncertainties)


def fit(
    x,
    y,
    dy,
    model,
    guess=None,
    bounds=None,
    method=None,
    error_method=1,
    respect_bounds=False,
    silent=False,
    pandas=True,
    **kwargs
):
    """
    Voert fit uit voor gegeven dataset en model

    Parameters
    ----------
    x : array_like
        x-waarden van datapunten
    y : array_like
        y-waarden van datapunten
    dy : array_like
        onzekerheid op y-waarden
    model : callable
        te fitten model. Eerste argument is x-waarde, andere de parameters
    guess : list
        startwaarden voor fitalgortime, lengte moet gelijk zijn aan aantal parameters
    bounds : list
        grenzen van de parameters, tuple van min en max voor elke parameter
    method : str
        algoritme gebruikt door scipy.optimize.minimize
    error_method : 1 | 2
        methode om fouten te bepalen
        1 : chikwadraat + 1
        2 : tweede afgeleide
    respect_bounds : bool
        True: gebruikt speciale gradient functie die model niet oproept met parameters buiten bounds
    silent : bool
        True: print geen p-waarden en andere
    pandas : bool
        True: geeft parameters en errors als pandas dataframe
        False: geeft parameters en errors als numpy matrix
    **kwargs
        worden doorgegven aan scipy.optimize.minimize
    """
    if type(dy) == int and dy == 0:
        dy = 1
        silent = True

    modelParams = getfullargspec(model).args
    if guess is None:
        guess = [1] * (len(modelParams) - 1)

    if len(modelParams) < len(guess):
        modelParams = list(map(chr, range(97, 123)))[: len(guess)]  # alfabet

    params = _fitParameters(
        x,
        y,
        dy,
        model,
        guess,
        bounds=bounds,
        method=method,
        respect_bounds=respect_bounds,
        **kwargs
    )["x"]
    if error_method == 1:
        errors = _errorFit(x, y, dy, model, params, symmetric=True)
    else:
        errors = _errorFit2(x, y, dy, model, params)

    if not silent:
        ls, rls, p = goodness_fit(x, y, dy, model, params)
        print("Least squares value: " + str(round(ls, 3)))
        print("Reduced Least squares: " + str(round(rls, 3)))
        print("p-value: " + str(round(p, 5)))

    arr = np.array([params, np.abs(errors)]).T
    if not pandas or not pandasGlobal:
        return arr
    else:
        return pd.DataFrame(arr, index=modelParams[1:], columns=["value", "error"])


def goodness_fit(x, y, dy, model, params):
    """Geeft chi2, reduced chi2 en p-waarde terug"""
    ls = chi2(x, y, dy, model, params)
    if len(y.shape) == 1:
        df = len(y) - len(params)
    else:
        df = np.prod(y.shape) - len(params)
    p = 1 - stats.chi2.cdf(ls, df)
    return ls, ls / df, p
