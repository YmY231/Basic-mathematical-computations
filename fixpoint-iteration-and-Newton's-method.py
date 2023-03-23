from array import array
import numpy as np
import matplotlib.pyplot as plt


def bisection(f,a,b,Nmax):
    
    """
    Bisection Method: Returns a numpy array of the 
    sequence of approximations obtained by the bisection method.
    
    Parameters
    ----------
    f : function
        Input function for which the zero is to be found.
    a : real number
        Left side of interval.
    b : real number
        Right side of interval.
    Nmax : integer
        Number of iterations to be performed.
        
    Returns
    -------
    p_array : numpy.ndarray, shape (Nmax,)
        Array containing the sequence of approximations.
    """
    
    # Initialise the array with zeros
    p_array = np.zeros(Nmax)
    
    # Continue here:...

    
    
    
    

    return p_array


def fixedpoint_iteration(g, p0, Nmax):
    """ 
    Fixed-point Iteration: Returns a numpy array of the 
    sequence of approximations obtained by the fixed-point iteration algorithm.
    
    Parameters 
    ----------
    g : function
        Input function for where the iteration is to be applied.
    p0 : real number
        The initial value of the iteration.
    Nmax : integer
        The maximum number of iterations to be performed.
    
    Returns 
    -------
    p_array : numpy.ndarray, shape (Nmax,)
        Array containing the sequence of approximations.
    """
    
    p_array = np.zeros(Nmax)
    i = 0
    while i < Nmax:
        p = g(p0)
        p_array[i] = p
        p0 = p
        i += 1
    
    return p_array


def fixedpoint_iteration_stop(g,p0 ,Nmax ,TOL):
    """ 
    fixed-point iteration method with a stopping criterion: Returns a numpy array of the 
    sequence of approximations obtained by the fixed-point iteration algorithm with a stopping criterion.
    
    Parameters 
    ----------
    g : function
        Input function for where the iteration is to be applied.
    p0 : real number
        The initial value of the iteration.
    Nmax : integer
        The maximum number of iterations to be performed.
    TOL : real number
        Value of the tolerant error.
        
    Returns 
    -------
    p_array : numpy.ndarray, shape (Nmax,)
        Array containing the sequence of approximations.
    """
    
    p_array = []
    i = 0
    while i < Nmax:
        p = g(p0)
        if abs(p - p0) <= TOL:
            break
        p_array.append(p)
        p0 = p
        i += 1
    return np.array(p_array)


def newton_stop(f, dfdx, p0, Nmax, TOL):
    """ 
    Newton's Method: Returns a numpy array of the 
    sequence of approximations obtained by the Newton's Method with a stopping criterion.
    
    Parameters 
    ----------
    f : function 
         Input function for where the Newton's Method is to be applied.
    dfdx : function 
        First derivative of the function f.
    p0 : real number 
        The initial value of the iteration.
    Nmax : real number 
        The maximum number of iterations to be performed.
    TOL : real number 
        Value of the tolerant error.
        
    Returns
    -------
    p_array : numpy.ndarray, shape (Nmax,)
        Array containing the sequence of approximations.
    """
    
    p_array = []
    i = 1
    while i < Nmax:
        p = p0 - (f(p0)/dfdx(p0))
        p_array.append(p)
        if abs(p - p0) <= TOL:
            break
        p0 = p
        i += 1
    return np.array(p_array)
    
    
def plot_convergence(p, f, dfdx, g, p0, Nmax):
    
    # Fixed -point iteration
    p_array = fixedpoint_iteration(g,p0,Nmax)
    e_array = np.abs(p - p_array)
    n_array = 1 + np.arange(np.shape(p_array)[0])
    
    # Newton method
    p_array_1 = newton_stop(f, dfdx, p0, Nmax, 10**(-16))
    print(p_array_1)
    e_array_1 = np.abs(p - p_array_1)
    n_array_1 = 1 + np.arange(np.shape(p_array_1)[0])

    # Preparing figure , using Object -Oriented (OO) style; see:
    # https :// matplotlib.org/stable/tutorials/introductory/quick_start.html
    fig, ax = plt.subplots()
    ax.set_yscale("log")
    ax.set_xlabel("n")
    ax.set_ylabel("|p-p_n|")
    ax.set_title("Convergence behaviour")
    ax.grid(True)
    
    # Plot
    ax.plot(n_array, e_array, 'o', label='FP iteration ', linestyle='--')
    ax.plot(n_array_1, e_array_1, 'p', label='NM iteration ', linestyle='-.')
    
    # Add legend
    ax.legend ();
    
    return fig, ax


def optimize_FPmethod(f, c_array, p0, TOL):
    
    C_N = []
    p_initial = p0
    for c in c_array:
        j = 0
        p0 = p_initial
        while j < 100:
            p = p0 - c * f(p0)
            if abs(p - p0) <= TOL:
                C_N.append([c, j+1])
                break
            p0 = p
            j += 1
    
    C_N   = np.array(C_N)
    indx  = np.argmin(C_N, axis=0)[1]
    c_opt = C_N[indx][0]
    n_opt = C_N[indx][1]
    
    return c_opt, n_opt
