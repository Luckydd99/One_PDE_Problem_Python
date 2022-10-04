"""
Luca Di Domenico, 233209.
Scientific computing, project 3, May-June 2022.
The functions below implement multigrid methods.
"""

import numpy as np

"""--------_________________________-----------
-----------|Multigrid in 1d: Task 1|-----------
-----------^^^^^^^^^^^^^^^^^^^^^^^^^--------"""

def gs_step_1d(uh,fh):
    """It executes one step of the Gauss-Siedel method for a 1d problem,
    it updates uh in situ, it returns the pseudo-residual. WE ASSUME THAT
    THE BOUNDARY CONDITIONS ARE ALREADY SATISFIED INSIDE THE INPUTS."""
    
    #Setup.
    N = len(uh)-1
    hSquared = (4/N)**2
    denom = (2+hSquared)
    
    #This is used for the pseudo-residual computation.
    olduh = uh.copy()
    
    #Simple form of the Gauss-Siedel method
    #(here c_i = 1 for all i; also uh[i-1] is updated
    #in the previous cycle thus uh[i-1] = u^(k+1)_{i-1}.
    for i in range(1,N):
        uh[i] = (hSquared*fh[i] + uh[i-1] + uh[i+1])/denom
    
    #The pseudo-residual is computed with the infinity norm.
    return max(np.abs(uh - olduh))

"""--------_________________________-----------
-----------|Multigrid in 1d: Task 4|-----------
-----------^^^^^^^^^^^^^^^^^^^^^^^^^--------"""

#v,w must be inteded as vectors s.t. len(v) = N+1, len(w) = (N//2)+1.
#halfN is the value len(w)-1 = N//2.

def restriction(v, halfN):
    """It returns the weighted restriction of v into a coarser 1d grid."""
    w = np.zeros(halfN + 1)
    for i in range(1,halfN):
        w[i] = (v[(2*i) - 1] + (2*(v[2*i])) + v[(2*i) + 1])/4
    return w

#<:-:-:-:-:-:-:-:-:::---:::-:-:-:-:-:-:-:-:>

def prolongation(w, N, halfN):
    """It returns the prolongation of w into a finer 1d grid."""
    v = np.zeros(N+1)
    for i in range(halfN):
        v[2*i] = w[i]
        v[(2*i) + 1] = (w[i] + w[i+1])/2
    v[-1] = w[-1]
    return v

#<:-:-:-:-:-:-:-:-:::---:::-:-:-:-:-:-:-:-:>

def compute_residual(uh,fh,N):
    """It returns the vector obtained by subtracting to fh the multiplication
    between the matrix hat(A_h) (which accounts for the inhomogeneous
    case) and the vector u_h associated with our 1d boundary value problem."""
    hSquared = (4/N)**2
    residual = np.zeros(N+1)
    #Note that in the boundary points fh[i] - 1*uh[i] = 0.
    for i in range(1, N):
        residual[i] = fh[i] - (uh[i] + ((2*uh[i]) - uh[i-1] - uh[i+1])/hSquared)
    return residual

#<:-:-:-:-:-:-:-:-:::---:::-:-:-:-:-:-:-:-:>

def v_cycle_step_1d(uh,fh,alpha1,alpha2):
    """It executes one step of the V-cycle method for a 1d problem,
    it updates uh in situ, it returns the pseudo-residual."""
    
    #Setup
    N = len(uh)-1
    
    if N == 2:
        #The exact solve on the trivial 3-point grid Omega_1.
        #NOTE!!! If we assume that the trivial grid Omega_1
        #is never passed as the first input of the v_cycle_step_1d,
        #then we can substitute the following four lines inside this
        #"if" case with these two lines:
        #uh[1] = (2/3)*fh[1]
        #return 0
        uh[0] = fh[0]
        uh[2] = fh[2]
        uh[1] = (fh[0]+fh[2])/6 + (2/3)*fh[1]
        return 0
    else:
        #Pre-smoothing.
        for _ in range(alpha1): gs_step_1d(uh, fh)
        
        #Restriction of the residual.
        halfN = N // 2
        r_2h = restriction(compute_residual(uh, fh, N), halfN)
        
        #Recursive call.
        errorVec = np.zeros(halfN+1)
        v_cycle_step_1d(errorVec, r_2h, alpha1, alpha2)
        
        #Error prolongation and coarse grid correction.
        uh[:] = uh + prolongation(errorVec, N, halfN)
        
        #Post-smoothing.
        for _ in range(alpha2): pseudo_residual = gs_step_1d(uh, fh)
        return pseudo_residual

"""--------_________________________-----------
-----------|Multigrid in 1d: Task 6|-----------
-----------^^^^^^^^^^^^^^^^^^^^^^^^^--------"""

#v,w must be inteded as vectors s.t. len(v) = N+1, len(w) = (N//2)+1.
#halfN is the value len(w)-1 = N//2.

def natural_restriction(v, halfN):
    """It returns the natural restriction of v into a coarser 1d grid."""
    w = np.zeros(halfN + 1)
    for i in range(halfN+1):
        w[i] = v[2*i]
    return w

#<:-:-:-:-:-:-:-:-:::---:::-:-:-:-:-:-:-:-:>

def full_mg_1d(uh, fh, alpha1, alpha2, nu):
    """It implements the full multigrid method for a 1d problem,
    it updates uh in situ, it returns the pseudo-residual."""
    
    #Setup.
    N = len(uh)-1
    
    if N == 2:
        #The exact solve on the trivial 3-point grid Omega_1.
        #The remark written in the solve of the trivial case
        #of v_cycle_step_1d still holds here.
        uh[0] = fh[0]
        uh[2] = fh[2]
        uh[1] = (fh[0]+fh[2])/6 + (2/3)*fh[1]
        return 0
    else:
        #Natural restriction of fh.
        halfN = N//2
        f_2h = natural_restriction(fh, halfN)
        
        #Recursive call.
        uTwoH = np.zeros(halfN+1)
        full_mg_1d(uTwoH, f_2h, alpha1, alpha2, nu)
        
        #Improved initial guess.
        uh[:] = prolongation(uTwoH,N,halfN)
        
        #V-cycle call(s).
        for _ in range(nu): pseudo_residual = v_cycle_step_1d(uh, fh, alpha1, alpha2)
        return pseudo_residual

"""--------_________________________________-----------
-----------|Multigrid in 2d: Task 1, Item 1|-----------
-----------^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^--------"""

def gs_step_2d(uh, fh):
    """It executes one step of the Gauss-Siedel method for a 2d problem,
    it updates uh in situ, it returns the pseudo-residual.
    uh is intended as a (N+1)X(N+1) array, such that uh[i,j]
    estimates the value of u in (x_i,x_j). WE ASSUME THAT
    THE BOUNDARY CONDITIONS ARE ALREADY SATISFIED INSIDE THE INPUTS."""
    
    #Setup.
    N = (np.shape(uh)[0])-1
    alpha = 4/((N**2)+4)
    beta = (N**2)/16
    
    #This is used for the pseudo-residual computation.
    originalUh = uh.copy()
    
    #Simplified form of the Gauss-Siedel method.
    #Indeed, u[i,j] = (f[i,j] + (1/h**2)*(uh[i,j-1] + uh[i-1,j]) +
    #(1/h**2)*(uh[i+1,j] + uh[i,j+1]))/((4/h**2) + 1) where h**2 = 16/(N**2)
    #and 1 = c(x_i,x_j) for all (x_i,x_j) in the 2d grid.
    #Note that uh[i,j-1] = u^(k+1)_{i,j-1} and uh[i-1,j]= u^(k+1)_{i-1,j}.
    for i in range(1,N):
        for j in range(1,N):
            uh[i,j] = alpha*(fh[i,j] + beta*(uh[i-1,j]
                        + uh[i,j-1] + uh[i+1,j] + uh[i,j+1]))
    
    #The pseudo-residual is computed with the infinity norm.
    return max(np.abs((uh-originalUh).flat))

"""--------_________________________________-----------
-----------|Multigrid in 2d: Task 1, Item 2|-----------
-----------^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^--------"""

#v,w must be inteded as bidimensional arrays s.t.
#(np.shape(v)[0]) = N+1, (np.shape(w)[0]) = (N//2)+1.
#halfN is the value (np.shape(w)[0])-1 = N//2.

def restriction_2d(v,halfN):
    """It returns the weighted restriction of v into a coarser 2d grid."""
    w = np.zeros((halfN+1,halfN+1))
    for i in range(1,halfN):
        for j in range(1,halfN):
            w[i,j] = (v[2*i-1, 2*j-1] + v[2*i-1, 2*j+1] + v[2*i+1, 2*j-1] + 
                      v[2*i+1, 2*j+1])/16 + (v[2*i, 2*j-1] + v[2*i, 2*j+1] + 
                      v[2*i-1, 2*j] + v[2*i+1, 2*j])/8 + v[2*i, 2*j]/4
    return w

#<:-:-:-:-:-:-:-:-:::---:::-:-:-:-:-:-:-:-:>


def prolongation_2d(w,N,halfN):
    """It returns the prolongation of w into a finer 2d grid."""
    v = np.zeros((N+1,N+1))
    #Since we always prolongate the error vector (which contains zeros
    #in the components associated with the boundary points), then the
    #prolongation v has zeros in v[0,:], v[:,0], v[-1,:] and v[:,-1].
    #So, these lines (with very few "wasted computations") suffice.
    for i in range(halfN):
        for j in range(halfN):
            v[2*i,2*j] = w[i,j]
            v[2*i+1,2*j] = (w[i,j] + w[i+1,j])/2
            v[2*i,2*j+1] = (w[i,j] + w[i,j+1])/2
            v[2*i+1, 2*j+1] = (w[i,j] + w[i+1,j] + w[i,j+1] + w[i+1,j+1])/4
    return v

#<:-:-:-:-:-:-:-:-:::---:::-:-:-:-:-:-:-:-:>

def compute_residual_2d(uh, fh, N):
    """It returns the vector (seen as a bidimensional array)
    obtained by subtracting to fh the multiplication between the matrix
    hat(A_h) (which accounts for the inhomogeneous case) and the vector
    u_h associated with our 2d boundary value problem."""
    #hat(A_h) in 2d is a ((N+1)**2)X((N+1)**2) pentadiagonal block matrix.
    #It becomes an identity matrix when we restrict it to the components
    #referred to the boundary points. Thus, at the boundary points,
    #fh - Ah*uh yield 0.
    hSquared = (4/N)**2
    r = np.zeros((N+1,N+1))
    for i in range(1, N):
        for j in range(1,N):
            r[i,j] = fh[i,j] - (uh[i,j] + ((4*uh[i,j]) - uh[i,j-1] -
                                       uh[i,j+1] - uh[i-1,j] - uh[i+1,j])/hSquared)
    return r
    
#<:-:-:-:-:-:-:-:-:::---:::-:-:-:-:-:-:-:-:>

def v_cycle_step_2d(uh, fh, alpha1, alpha2):
    """It executes one step of the V-cycle method for a 2d problem,
    it updates uh in situ, it returns the pseudo-residual."""
    
    #Setup
    N = (np.shape(uh)[0])-1
    
    if N == 2:
        #The exact solve on the trivial (3x3)-point grid Omega_1.
        #NOTE!!! If we assume that the trivial grid Omega_1
        #is never passed as the first input of the v_cycle_step_2d,
        #then we can substitute the following six lines inside this
        #"if" case with these two lines:
        #uh[1,1] = fh[1,1]/2
        #return 0
        uh[0,:] = fh[0,:]
        uh[2,:] = fh[2,:]
        uh[1,0] = fh[1,0]
        uh[1,2] = fh[1,2]
        uh[1,1] = (fh[1,0] + fh[1,2] + fh[0,1] + fh[2,1])/8 + (fh[1,1])/2
        return 0
    else:
        #Pre-smoothing.
        for _ in range(alpha1): gs_step_2d(uh, fh)
        
        #Restriction of the residual.
        halfN = N // 2
        r_2h = restriction_2d(compute_residual_2d(uh, fh, N), halfN)
        
        #Recursive call.
        errorVec = np.zeros((halfN+1,halfN+1))
        v_cycle_step_2d(errorVec, r_2h, alpha1, alpha2)
        
        #Error prolongation and coarse grid correction.
        uh[:,:] = uh + prolongation_2d(errorVec, N, halfN)
        
        #Post-smoothing.
        for _ in range(alpha2): pseudo_residual = gs_step_2d(uh, fh)
        return pseudo_residual

"""--------_________________________________-----------
-----------|Multigrid in 2d: Task 1, Item 3|-----------
-----------^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^--------"""

#v,w must be inteded as bidimensional arrays s.t.
#(np.shape(v)[0]) = N+1, (np.shape(w)[0]) = (N//2)+1.
#halfN is the value (np.shape(w)[0])-1 = N//2.

def func_g_2d(x,y):
    """It returns the value of g(x,y). It fixes the values of the 
    2d soultion uh in the boundary points."""
    return (x**2 + y**2) / 10

#<:-:-:-:-:-:-:-:-:::---:::-:-:-:-:-:-:-:-:>

def natural_restriction_2d(v,halfN):
    """It returns the natural restriction of v into a coarser 2d grid."""
    w = np.zeros((halfN+1,halfN+1))
    for i in range(halfN+1):
        for j in range(halfN+1):
            w[i,j] = v[(2*i), (2*j)]
    return w

#<:-:-:-:-:-:-:-:-:::---:::-:-:-:-:-:-:-:-:>

def prolongation_boundary(w, N, halfN):
    """It returns the prolongation v of w into a finer 2d grid, where the
    non-trivial values in the boundary points are preserved."""
    
    #Setup.
    v = np.zeros((N+1,N+1))
    
    #We define the values n the components of v
    #associated to the inner points (with very few "wasted computations").
    for i in range(halfN):
        for j in range(halfN):
            v[2*i,2*j] = w[i,j]
            v[2*i+1,2*j] = (w[i,j] + w[i+1,j])/2
            v[2*i,2*j+1] = (w[i,j] + w[i,j+1])/2
            v[2*i+1, 2*j+1] = (w[i,j] + w[i+1,j] + w[i,j+1] + w[i+1,j+1])/4
    
    #We define the values in the components of v
    #associated to the boundary points.
    h = 4/N
    for k in range(N+1):
        coord = -2+(k*h)
        v[0,k] = func_g_2d(-2, coord)
        v[-1,k] = func_g_2d(2, coord)
        v[k,0] = func_g_2d(coord, -2)
        v[k,-1] = func_g_2d(coord, 2)
    return v

#<:-:-:-:-:-:-:-:-:::---:::-:-:-:-:-:-:-:-:>

def full_mg_2d(uh, fh, alpha1, alpha2, nu):
    """It implements the full multigrid method for a 2d problem,
    it updates uh in situ, it returns the pseudo-residual."""
    
    #Setup.
    N = (np.shape(uh)[0])-1
    
    if N == 2:
        #The exact solve on the trivial (3x3)-point grid Omega_1.
        #The remark written in the solve of the trivial case
        #of v_cycle_step_2d still holds here.
        uh[0,:] = fh[0,:]
        uh[2,:] = fh[2,:]
        uh[1,0] = fh[1,0]
        uh[1,2] = fh[1,2]
        uh[1,1] = (fh[1,0] + fh[1,2] + fh[0,1] + fh[2,1])/8 + (fh[1,1])/2
        return 0
    else:
        #Natural restriction of fh.
        halfN = N//2
        f_2h = natural_restriction_2d(fh, halfN)
        
        #Recursive call.
        u_2h = np.zeros((halfN+1,halfN+1))
        full_mg_2d(u_2h, f_2h, alpha1, alpha2, nu)
        
        #Improved initial guess.
        uh[:] = prolongation_boundary(u_2h, N, halfN)
        
        #V-cycle call(s).
        for _ in range(nu): res = v_cycle_step_2d(uh, fh, alpha1, alpha2)
        return res

#<---__------___-----___----->
#<---||------||\\----||\\---->
#<---||------||-||---||-||--->
#<---||------||-||---||-||--->
#<---||___---||_//---||_//--->
#<---|____|--|__/----|__/---->