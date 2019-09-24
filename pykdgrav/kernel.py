from numba import njit, float64, float32

@njit(fastmath=True)#([float64(float64,float64),float32(float32,float32)])
def ForceKernel(r, h):
    """
    Returns the quantity (fraction of mass enclosed)/ r^3 for a cubic-spline mass distribution of compact support radius h. Used to calculate the softened gravitational force.

    Arguments:
    r - radius
    h - softening     
    """
    if r > h: return 1./(r*r*r)
    hinv = 1./h
    q = r*hinv
    if q <= 0.5:
        return (10.666666666666666666 + q*q*(-38.4 + 32.*q))*hinv*hinv*hinv
    else:
        return (21.333333333333 - 48.0 * q + 38.4 * q * q - 10.666666666667 * q * q * q - 0.066666666667 / (q * q * q))*hinv*hinv*hinv

@njit(fastmath=True)#([float64(float64,float64)])
def PotentialKernel(r, h):
    """
    Returns the equivalent of -1/r for a cubic-spline mass distribution of compact support radius h. Used to calculate the softened gravitational potential.

    Arguments:
    r - radius
    h - softening     
    """    
    if h==0.:
        return -1./r            
    hinv = 1./h
    q = r*hinv
    if q <= 0.5:
         return (-2.8 + q*q*(5.33333333333333333 + q*q*(6.4*q - 9.6))) * hinv
    elif q <= 1:
        return (-3.2 + 0.066666666666666666666 / q + q*q*(10.666666666666666666666 +  q*(-16.0 + q*(9.6 - 2.1333333333333333333333 * q)))) * hinv
    else:
        return -1./r