import numpy as np
import matplotlib.pyplot as plt
import warmup_solution as ws



def find_max(M,s,n,i):
    m = 0
    p = 0
    for j in range(i,n):
        if abs(M[j,i]/s[j]) > m:
            m = abs(M[j,i]/s[j])
            p = j
    
    return p



def scaled_partial_pivoting(A,b,n,c):
    
    # Compute the scale factors
    s = np.amax(np.abs(A),1)
    # Obtain the augmented matrix
    M = np.c_[A, b]
    # Interchange rows
    for i in range(c):
        p = find_max(A, s, n, i)
        M[[p,i],:] = M[[i,p],:]
        # Elimination
        for j in range(i+1,n):
            m = M[j,i]/M[i,i]
            for k in range(i,n+1):
                M[j,k] = M[j,k] - m*M[i,k]

    return M



def spp_solve(A,b,n):
    
    M = scaled_partial_pivoting(A,b,n,n-1)
    x = ws.backward_substitution(M,n)
    
    return x



def PLU(A,n):
    
    # Set P=I
    P = np.identity(n)
    # Set L to be a zero matrix
    L = np.zeros([n,n])
    # Set U=A
    U = A.copy()
    
    for i in range(n-1):
        for s in range(i,n):
            if abs(U[s,i]) > 1e-15: break
        if s != i:
            P[[i,s],:] = P[[s,i],:]
            L[[i,s],:] = L[[s,i],:]
            U[[i,s],:] = U[[s,i],:]
        for j in range(i+1,n):
            L[j,i] = U[j,i]/U[i,i]
            U[j,i] = 0
            for k in range(i+1,n):
                U[j,k] = U[j,k] - L[j,i] * U[i,k]
    
    P = np.transpose(P)
    L = L + np.identity(n)
    
    
    return P, L, U



def Jacobi(A,b,n,x0,N):
    
    x_approx = np.tile(x0, 1)
    
    for k in range(N):
        X = np.tile(x0, 1)
        x = np.repeat(x_approx[:,k], 1, axis=0)
        
        for i in range(n):
            X[i] = (1/A[i,i]) * sum([-A[i,j]*x[j] for j in range(n) if j != i]) + b[i]/A[i,i]
        
        x_approx = np.c_[x_approx, X]
        
    return x_approx



def Jacobi_plot(A,b,n,x0,N):

    # Create array of k values
    k_array = np.arange(N+1)
    # Prepare figure
    fig, ax = plt.subplots()
    ax.set_yscale("log")
    ax.set_xlabel("$k$")
    ax.grid(True)
    
    x       = ws.no_pivoting_solve(A,b,n)[:,-1]
    x_k     = Jacobi(A,b,n,x0,N)
    error_i = np.zeros(N+1)
    error_2 = np.zeros(N+1)
    
    for k in range(N+1):
        error_i[k] = np.linalg.norm((x-x_k[:,k]),np.inf)
        error_2[k] = np.linalg.norm((x-x_k[:,k]),2)
    
    # Plot
    ax.plot(k_array,error_i,"s",label="$||x-x^{(k)}||_\infty$")
    ax.plot(k_array,error_2,"o",label="$||x-x^{(k)}||_2$")
    # Add legend
    ax.legend()
    return fig, ax
