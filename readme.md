# FysFuncties

Bevat nuttige functies voor practica.

## Installeren

    pip install fysfuncties

# Voorbeelden

```python
import fysfuncties as ff
import sympy as sp
import numpy as np
import pandas as pd
```
## Error propagatie

functie f = x*sqrt(y) met


x = 1.8 +/- 0.2

y = 56.0  +/- 1.2
```python
# definieer sympy symbolen en functie
x, y = sp.symbols("x y")
f = x * sp.sqrt(y)
waarde, fout = ff.errorprop(f, [x, y], [1.8, 56.0], [0.2, 1.2])
>>> 13.4699665923862, 1.50360519134132
```
## Fitten

## Afronden
