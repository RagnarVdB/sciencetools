import numpy as np
import sympy as sp
from scipy.misc import derivative
from scipy import stats


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