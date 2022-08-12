This is a matrix multiplication problem you are trying to solved. Only when the number of rows in the second matrix is equals to the number of columns in the first matrix can two matrices be multiplied. Therefore we need to change the shape of the second variable to mach with the first variable
"""

import numpy as np

a = np.array([[-1,2,3]])
b = np.array([[0,2,1]])
np.dot(a,b)

"""below is the code to check the shape of the variables in matrix"""

a.ndim

b.ndim

"""The two variables, a and b, both have the same dimension when you look at their shapes.
In order to resolve the issue, you must modify the second matrix, which happens to be b, as seen below.
"""

b = b.reshape(3, 1)
b

"""We will now run the code BELOW to examine the results after changing the form of the variable."""

np.dot(a,b)
