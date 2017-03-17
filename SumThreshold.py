from numpy import copy, delete,array

def SumThreshold(x,y,M,chi):

#x (float array) is the data itself (1D)
#y (binary [coded as int] array) is the previous flag mask - all zeros if no flagging has been done
#M (int) is the desired subsequence
#chi (float) is the threshold (regarded as a function of M in the paper)


    N = len(x)

    #These are for use in the loops, t will be the new flag mask
    z = 0
    q = 0
    count = 0
    t = copy(y)

    #This loop creates the window
    while q < M:
        if y[q] == 0:
            z += x[q]
            count += 1
        q += 1

    #This loop slides the window
    while q < N:
        if abs(z) > count*chi:
            t[q-M:q] = ones(M,int) #Flag subsequence of length M if exceeds threshold
        if y[q] == 0:
            z += x[q]
            count += 1
        if y[q-M] == 0:
            z -= x[q-M]
            count -= 1
        q += 1

    return(t)

    
