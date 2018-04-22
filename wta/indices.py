'''
Created on Nov 21, 2017

Takes a call-able parameter which will be used in the condition part of the 
list comprehension. Use a lambda or other function object to pass your arbitrary 
condition:

Example:
a = [1, 2, 3, 1, 2, 3, 1, 2, 3]
inds = indices(a, lambda x: x > 2)

>>>inds
[2, 5, 8]

@author: mark.lambrecht
'''


def indices(a, func):
    return [i for (i, val) in enumerate(a) if func(val)]
