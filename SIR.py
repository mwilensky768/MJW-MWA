from numpy import zeros

def SIROperator(FMi,Agg): #Takes a flag mask ( COLUMN vector) and aggression param. (number) as input

    N = len(FMi)

    Psi = Fmi + Agg - 1 #Initialize psi (this is a temp. array)
    M = zeros([N,1],float) #Initialize M (this is also temp.)

    for n in range(0,N): #Define M as in AOFlagger paper
        M[n+1,0] = M[n,0] + Psi[n,0]

    P = zeros([N,1],int) #Initialize P - this is a temp. array which is to be constructed so that M(P(n)) = min M(i), 0 <= i <= n (perhaps to be called the "latest min")

    for n in range(1,N): #This loop is really clever - I probably wouldn't have come up with it
        P[n,0] = P[n-1,0] #RHS is the last minima
        if M[P[n,0],0] > M[n,0]: #Satisfaction of this is to say a new latest min has been found
            P[n,0] = n

    Q = zeros([N,1],int) #Similar to P, but looks for max M(j) x <= j <= N-1

    for n in range(1,N): #Similar loop as before - but has to count backwards
        Q[N-1-n,0] = Q[N-n,0]
        if M[Q[N-1-n,0],0] < M[N-n,0]:
            Q[N-1-n,0] = N-n

    FMf = zeros([N,1],int) #Initialize output flag mask

    for n in range(0,N): #Ask important flagging question
        if M[Q[n,0],0] - M[P[n,0],0] >= 0:
           Fmf[n,0] = 1
       
    return(Fmf)
