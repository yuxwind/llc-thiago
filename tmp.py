import numpy as np

# sample.py
#myGlobal = 5

def func1():
    global myGlobal
    myGlobal = 42
    print(myGlobal)

def func2():
   # global myGlobal
    myGlobal = 5
    print(myGlobal)

func1()
func2()
print(myGlobal)
