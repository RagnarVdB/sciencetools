import math
import sympy as sp
import numpy as np
import scipy.optimize as opt
from scipy.misc import derivative
from scipy import stats


def errorprop(f, variables, values, errors):
  returnvals = f.subs([(var, val) for var, val in zip(variables, values)])
  derivsum = sum([sp.Derivative(f, var)**2 * err**2 for var, err in zip(variables, errors)])
  prop = sp.sqrt(derivsum).doit().subs([(var, val) for var, val in zip(variables, values)])
  return sp.N(returnvals), sp.N(prop)


def chi2(x, y, dy, model, parameters):
    # Geeft chi^2 waarde voor gegeven parameters, model en waarden
    errors = ((y - model(x, *parameters))**2) / dy**2
    return np.sum(errors)

def fitParameters(x, y, dy, model, guess, bounds=None, method=None):
    # minimaliseerd de chi2 functie
    minobj = opt.minimize(lambda p: chi2(x, y, dy, model, p), guess, bounds=bounds, method=method)
    #if minobj["success"]:
        #print("success")
    #else:
        #print("unsuccessful")
        #print(minobj["message"])
    return(minobj["x"])

def substitute(x, i, array):
    # Vervangt één element in een numpy array
    new_array = np.array(array)
    new_array[i] = x
    return new_array

def intersect(x_array, y_array1, y_array2):
    # Geeft intersects van twee numpy arrays
    intersect_indices = np.argwhere(np.diff(np.sign(y_array1 - y_array2))).flatten()
    intersects = x_array[intersect_indices]
    return intersects

def errorFit2(x, y, dy, model, minimum):
    # Geeft onzekerheidsinterval rond fitparameters
    uncertainties = []
    for i in range(len(minimum)):
        deriv = derivative(lambda p: chi2(x, y, dy, model, substitute(p, i, minimum)), minimum[i], n=2)
        uncertainties.append(np.sqrt(2/deriv))
    return np.array(uncertainties)

def min_chi2(xy, chi2_single, minimum):
    x, y = xy
    return np.array([y - chi2_single(x), y - minimum - 1])


def errorFit(x, y, dy, model, minimum, symmetric=False):
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

def fit(x, y, dy, model, guess, bounds=None, method=None, error_method=1, silent=False):
    if dy == 0:
        dy = 1
        silent=True
    params = fitParameters(x, y, dy, model, guess, bounds=bounds, method=method)
    if error_method == 1:
        errors = errorFit(x, y, dy, model, params, symmetric=True)
    else:
        errors = errorFit2(x, y, dy, model, params)
    if not silent:
        ls = chi2(x, y, dy, model, params)
        df = len(x) - len(guess)
        print("Least squares value: " + str(round(ls, 3)))
        print("Reduced Least squares: " + str(round(ls/df, 3)))
        p = 1 - stats.chi2.cdf(ls, df)
        print("p-value: " + str(round(p, 3)))
    return np.array([params, np.abs(errors)]).T


def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx

def weighted_average(waardenlijst, foutenlijst):
    #returnt gewogen gemiddelde en de bijhorende fout
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

def get_powers(value, error):
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
    value = float(value)
    error = float(error)
    if precision == None or power_error == None or precision_error == None:
        precision, power_error, precision_error = get_powers(value, error)
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
    if type(errors) == float and type(values) != float:
        errors = [error]*len(values)
    precisions, power_errors, precision_errors = np.array([get_powers(value, error) for value, error in zip(values, errors)]).T
    if power == None:
        if (power_errors < 0).any():
            power = -min(power_errors)
        else:
            power = 0
    strings = []
    for i in range(len(values)):
        strings.append(rounder(values[i], errors[i], power, precisions[i], power_errors[i], precision_errors[i], False))
    return np.array(strings, dtype=str), power
