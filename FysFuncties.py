import math
import sympy as sp
import numpy as np
import pandas as pd
import scipy.optimize as opt
from scipy.misc import derivative
from scipy import stats
from inspect import getfullargspec

pandasGlobal = True

def set_pandas(pandas):
    pandasGlobal = pandas

def errorprop(f, variables, values, errors):
    """Propagate errors"""
    returnvals = f.subs([(var, val) for var, val in zip(variables, values)])
    derivsum = sum([sp.Derivative(f, var)**2 * err**2 for var, err in zip(variables, errors)])
    prop = sp.sqrt(derivsum).doit().subs([(var, val) for var, val in zip(variables, values)])
    return np.float64(sp.N(returnvals)), np.float64(sp.N(prop))

def errorprop2(f, values, errors):
    value = f(*values)
    som = 0
    for i in range(0, len(values)):
        deriv = derivative(lambda x: f(*values[0:i], x, *values[i+1:len(values)]), values[i])
        som += (deriv**2 * errors[i]**2)
    error = np.sqrt(som)
    return (value, error)

def chi2(x, y, dy, model, parameters):
    """Geeft least-squares waarde terug"""
    errors = ((y - model(x, *parameters))**2) / dy**2
    return np.sum(errors)

def _fitParameters(x, y, dy, model, guess, bounds=None, method=None):
    """minimaliseerd de chi2 functie"""
    minobj = opt.minimize(lambda p: chi2(x, y, dy, model, p), guess, bounds=bounds, method=method)
    return(minobj["x"])

def substitute(x, index, array):
    """Vervangt één element in een numpy array"""
    new_array = np.array(array)
    new_array[index] = x
    return new_array

def intersect(x_array, y_array1, y_array2):
    """Geeft intersects van twee numpy arrays"""
    intersect_indices = np.argwhere(np.diff(np.sign(y_array1 - y_array2))).flatten()
    intersects = x_array[intersect_indices]
    return intersects

def _errorFit2(x, y, dy, model, minimum):
    """Geeft onzekerheidsinterval rond fitparameters via tweede afgeleide"""
    uncertainties = []
    for i in range(len(minimum)):
        deriv = derivative(lambda p: chi2(x, y, dy, model, substitute(p, i, minimum)), minimum[i], n=2)
        uncertainties.append(np.sqrt(2/deriv))
    return np.array(uncertainties)

def _min_chi2(xy, chi2_single, minimum):
    x, y = xy
    return np.array([y - chi2_single(x), y - minimum - 1])


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

def fit(x, y, dy, model, guess=None, bounds=None, method=None, error_method=1, silent=False, pandas=True):
    """Voert fit uit voor gegeven dataset en model"""
    if type(dy) == int and dy == 0:
        dy = 1
        silent=True
    
    modelParams = getfullargspec(model).args
    if guess is None:
        guess = [1]*(len(modelParams) - 1)

    params = _fitParameters(x, y, dy, model, guess, bounds=bounds, method=method)
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

def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx

def weighted_average(waardenlijst, foutenlijst):
    """Geeft gewogen gemiddelde en de bijhorende fout"""
    gewichtenlijst = []
    for fout in foutenlijst:
        gewichtenlijst.append(1 / (fout**2))
    fout_op_gemiddelde = 1 / np.sqrt(sum(gewichtenlijst))

    tellerlijst = []
    for i in range(len(waardenlijst)):
        tellerlijst.append(gewichtenlijst[i]*waardenlijst[i])

    teller = sum (tellerlijst)
    noemer = sum (gewichtenlijst)

    gewogen_gemiddelde = teller / noemer
    # print("Het gewogen gemiddelde is ", gewogen_gemiddelde, "+-", fout_op_gemiddelde)
    return gewogen_gemiddelde, fout_op_gemiddelde

def find_zero_indices(array):
    diffs = (np.diff(np.sign(array)) != 0)
    indices = np.argwhere(diffs == True)[:,0]
    return indices

def _get_powers(value, error):
    power_error = -math.floor(math.log(error, 10))
    first_char = int(str(round(error*10**(power_error), 1))[0])
    power_error += int(first_char == 1)
    #print(power_error)

    if "." in str(value):
        precision = len(str(value).split('.')[1])
    else:
        precision = 0
    if "." in str(error):
        precision_error = len(str(error).split('.')[1])
    else:
        precision_error = 0
    return precision, power_error, precision_error

def rounder(value, error, power=None, precision=None, power_error=None, precision_error=None, showPower=True):
    """Print waarde en fout in latex"""
    value = float(value)
    error = float(error)
    if precision == None or power_error == None or precision_error == None:
        precision, power_error, precision_error = _get_powers(value, error)
    diff_power = 0
    if power == None:
        if power_error < 0:
            power = -power_error
        else:
            power = 0
    else:
        if -power_error > power:
            diff_power = -power_error - power
            power = -power_error
    value_rounded = round(value * float(10) ** (-power), power_error + power)
    error_rounded = round(error * float(10) ** (-power), power_error + power)
    if power_error + power == 0:
        value_rounded = int(value_rounded)
        error_rounded = int(error_rounded)
    elif power_error + power < 0:
        raise "No power could be found"
    elif power_error + power > 0:
        if precision < power_error:
            deficit = power_error - precision
            value_rounded = str(value_rounded) + "0"*deficit
        if precision_error < power_error:
            deficit = power_error - precision_error
            error_rounded = str(error_rounded) + "0"*deficit

    value_rounded = str(value_rounded).replace(".", ",")
    error_rounded = str(error_rounded).replace(".", ",")

    if showPower and power != 0:
        if power == 1:
            return "$({0} \\pm {1}) \\cdot 10$".format(value_rounded, error_rounded)
        else:
            return "$({0} \\pm {1}) \\cdot 10^{{{2}}}$".format(value_rounded, error_rounded, power)
    elif (showPower and power == 0) or (not showPower and not diff_power):
        return "${0} \\pm {1}$".format(value_rounded, error_rounded)
    elif not showPower and diff_power:
        return "$({0} \\pm {1}) \\cdot 10^{{{2}}}$".format(value_rounded, error_rounded, diff_power)

def rounder_array(values, errors, power=None):
    """Geeft array van waarden en fouten in latex"""
    if type(errors) == float and type(values) != float:
        errors = [errors]*len(values)
    precisions, power_errors, precision_errors = np.array([_get_powers(value, error) for value, error in zip(values, errors)]).T
    if power == None:
        if (power_errors < 0).any():
            power = -min(power_errors)
        else:
            power = 0
    strings = []
    for i in range(len(values)):
        strings.append(rounder(values[i], errors[i], power, precisions[i], power_errors[i], precision_errors[i], False))
    return np.array(strings, dtype=str), power

def print_dataframe(df, values=None, errors=None, powers=None):
    """geeft pandas dataframe terug in latexstrings"""
    cols = df.columns
    if not values or not errors:
        values = []
        errors = []
        for col in cols:
            if col[0] == "d" and col[1:] in cols:
                values.append(col[1:])
                errors.append(col)
        for col in cols:
            if not col in values and not col in errors:
                values.append(col)
    if not powers:
        powers = [None]*len(errors)
    pd_dict = {}
    for i in range(len(values)):
        if i < len(errors):
            pd_dict[values[i]] = rounder_array(df[values[i]], df[errors[i]], powers[i])[0]
        else:
            if i < len(powers) and powers[i]:
                pd_dict[values[i]] = df[values[i]]*10**(-powers[i])
            else:
                pd_dict[values[i]] = df[values[i]]

    return pd.DataFrame(pd_dict)

def test_compatibiliteit(x, sx, y, sy, alpha=None):
    """Test compatibiliteit van twee waarden"""
    fout = np.sqrt(sx**2 + sy**2)
    z = np.abs(x-y) / fout
    p = 2*(1 - stats.norm.cdf(z))
    print("z-value is:  ", str(round(z, 3)))
    print("p-value is:  ", str(round(p, 3)), " = ", str(round(p*100,1)),"%")
    if alpha:
        if p < alpha:
            print("REJECT at alpha = {}".format(alpha))
        else:
            print("NOT REJECTED at alpha = {}".format(alpha))
    else:
        if p < 0.05:
            print("REJECT at alpha = 0.05")
        if p < 0.01:
            print("REJECT at alpha = 0.01")

    return z, p
