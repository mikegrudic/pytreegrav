from numba import njit, float64, float32

@njit([float64(float64,float64),float32(float32,float32)])
def ForceKernel(r, h):
    hinv = 1./h
    q = r*hinv
    if q <= 0.5:
        return (10.666666666666666666 + q*q*(-38.4 + 32.*q))*hinv*hinv*hinv
    elif 0.5 < q <= 1:
        return (21.333333333333 - 48.0 * q + 38.4 * q * q - 10.666666666667 * q * q * q - 0.066666666667 / (q * q * q))*hinv*hinv*hinv
    else:
        return 1./(r*r*r)

@njit([float64(float64,float64),float32(float32,float32)])
def PotentialKernel(r, h):
    hinv = 1./h
    q = r*hinv

    if q<= 0.5:
        return (-2.8 + q*q*(5.33333333333333333 + q*q*(6.4*q - 9.6))) * hinv
    elif 0.5 < q <= 1:
        return (-3.2 + 0.066666666666666666666 / q + q*q*(10.666666666666666666666 +  q*(-16.0 + q*(9.6 - 2.1333333333333333333333 * q)))) * hinv
    else:
        return -1./r
    