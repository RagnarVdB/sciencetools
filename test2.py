import numpy as np
import math


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
      return "(${0} \\pm {1}$) \\cdot 10".format(value_rounded, error_rounded)
    else:
      return "(${0} \\pm {1}$) \\cdot 10^{{{2}}}".format(value_rounded, error_rounded, power)
  elif (showPower and power == 0) or (not showPower and not diff_power):
    return "${0} \\pm {1}$".format(value_rounded, error_rounded)
  elif not showPower and diff_power:
    return "(${0} \\pm {1}$) \\cdot 10^{{{2}}}".format(value_rounded, error_rounded, diff_power)

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




values = np.array([10, 20, 30])
errors = np.array([1, 2, 3])


tests = np.array(
[[586.72922437, 5.98700235],
[558.49785319, 60.76305235],
[450.91124088, 40.34894643],
[449.23073674, 30.47774837],
[437.98889626, 20.06806906],
[428.29830438, 320.7114207 ],
[424.74543606, 1.524974886]])

strings, power = rounder_array(*tests.T)
print()
for string in strings:
  print(string)
print(power)




# TESTS

TESTS = [
    ((137.678, 0.38), "$137,7 \pm 0,3$"),
    ((1.7, 0.14), "$1,70 \pm 0,14$")
  ]

def test(params, returns):
  if (rounder(*params)) == returns:
    print("✔ Success! ")
  else:
    print("❌ failed ")
    print("geeft: " + rounder(*params))
    print("ipv  : " + returns)
def main():
  for t in TESTS:
    test(*t)

#main()