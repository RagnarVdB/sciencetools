import numpy as np
import pandas as pd
import math

from sciencetools.stats import errorprop

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

def _print_string(value, error, power, latex):
    if power is None:
        if latex:
            return f"${value} \\pm {error}$"
        else:
            return f"{value} ± {error}"
    elif power == 1:
        if latex:
            return f"$({value} \\pm {error}) \\cdot 10$"
        else:
            return f"({value} ± {error}) * 10"
    else:
        if latex:
            return f"$({value} \\pm {error}) \\cdot 10^{{{power}}}$"
        else:
            return f"({value} ± {error}) * 10^{power}"

def rounder(value, error, power=None, precision=None, power_error=None, precision_error=None, showPower=True, decimal=".", latex=True):
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

    value_rounded = str(value_rounded).replace(".", decimal)
    error_rounded = str(error_rounded).replace(".", decimal)

    if showPower and power != 0:
        return _print_string(value_rounded, error_rounded, power, latex)
    elif (showPower and power == 0) or (not showPower and not diff_power):
        return _print_string(value_rounded, error_rounded, None, latex)
    elif not showPower and diff_power:
        return _print_string(value_rounded, error_rounded, diff_power, latex)

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