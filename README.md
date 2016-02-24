# pychimerge
ChiMerge: Discretization of Numeric Attributes

Python implementation of ChiMerge, a bottom-up discretization method based on ChiSqrt test.
- Reference paper: https://www.aaai.org/Papers/AAAI/1992/AAAI92-019.pdf


# Examples

1. Database: IRIS, Attribute: 1 (Sepal Width)
    - Intervals   : [ 2.   2.5  3.   3.4]
    - Chi2: [0-1) :  4.7, [1-2): 17.1, [2-3): 24.2
    - Interval-Class Frequencies (4x3)
    - [[  1.   9.   1.]
      [  1.  25.  20.]
      [ 18.  15.  24.]
      [ 30.   1.   5.]]

2. Database: TOI Example
    - Intervals   : [ 1 11 45]
    - Chi2        : [0-1):  2.7, [1-2):  3.9
    - Interval-Class Frequencies (3x2)
    - [[ 4.  1.]
      [ 1.  3.]
      [ 3.  0.]]
