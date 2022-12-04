# from func import f

def f(x):
    return x * x

def Trap(a, b, n, h):
    integral = (f(a) + f(b)) / 2.0

    x = a
    for i in range(1, int(n)):
        x = x + h
        integral = integral + f(x)
    
    return integral * h