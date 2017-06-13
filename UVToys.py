import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pyuvdata as pyuv
from scipy.stats import rayleigh

OPTIONS = [' [All Data]',' [No Band Edges, Yes Time Edges]',' [Yes Band Edges, No Time Edges]',' [No Band/Time Edges]']

#selects indices to be used form OPTIONS list...
def Opt(x,y):
    if not x and not y:
        return(' [All Data]')
    elif x and not y:
        return(' [No Band Edges, Yes Time Edges]')
    elif not x and y:
        return(' [Yes Band Edges, No Time Edges]')
    elif x and y:
        return(' [No Band/Time Edges]')

def UVstats(X): #Takes an array and outputs its stat. quantities in another array (6)

    x = np.zeros(4)

    x[0] = np.mean(X)
    x[1] = np.var(X)
    x[2] = np.amax(X)
    x[3] = np.amin(X)

    return(x)

def VisPlot(X,m): #X is an Obs. ID (int), m is baseline (int) (17)

    UV = np.load('/Users/radcosgroup/UVFITS/' + str(X) +'D' + str(m) + '.npy')
    UVamp = abs(UV)

    stats = UVstats(UVamp)

    

    plt.imshow(UVamp, cmap = cm.coolwarm)
    plt.title(str(X) + ' Baseline ' + str(m) + ' XX Visibilities')
    plt.xlabel('Frequency Channel Number')
    plt.ylabel('Time From Beginning (2s)')
    plt.colorbar()

    plt.show()

def SubSequenceSum(x,M): #input data is x (1-d), M is subsequence length of interest (35)
    N = len(x) #Number of data points
    y = np.zeros(N-M+1, dtype = complex) #this will be returned - it is sum of subsequences

    for k in range(0,N-M+1): #assign values to y
        y[k] = np.sum(x[k:k+M]) #perform the sum

    return(y) #Return the sums in a numpy array

def FlagPlot(X,m): #Same info as VisPlot (44)

    FA = np.load('/Users/radcosgroup/UVFITS/' + str(X) +'F' + str(m) + '.npy')

    

    plt.imshow(FA, cmap = cm.binary)
    plt.title(str(X) + ' Baseline ' + str(m) + ' XX Flag')
    plt.xlabel('Frequency Channel Number')
    plt.ylabel('Time From Beginning (2s)')

    

    plt.show()

def ReadEvenOdd(filepath,m): #filepath = 'string', m = 1 => even, m = 0 => odd (58)

    UV = pyuv.UVData()
    UV.read_uvfits(filepath)

    UV.select(times = [UV.time_array[(2*k+m)*8128] for k in range(0,28)])
    #The first index in the array referred to in the list comprehension is always 67585
    return(UV)

def CoarseBandEdgeFilter(UV,m): #(UVData(),bool) True removes the coarse band edges!

    LEdges = [0+16*p for p in range(24)] #Coarse bands are 16 channels wide: 384/16 = 24
    REdges = [15+16*p for p in range(24)]

    if m:
        UV.select(freq_chans = [n for n in range(384) if n not in LEdges and n not in REdges])
    else:
        UV.select(freq_chans = [n for n in range(384) if n in LEdges or n in REdges])
    
    return(UV)

def EMO(filepath,BEFilter,TEFilter): #(UVData(), bool, bool) True removes data! This makes the histogram object (79)

    #Read in the even and odd times (user made functions based on Bryna's excellent select feature in pyuvdata)
    UVeven = ReadEvenOdd(filepath,1)
    UVodd = ReadEvenOdd(filepath,0)
    
    if BEFilter: #This removes the data in the coarse band edges
        UVeven = CoarseBandEdgeFilter(UVeven,True)
        UVodd = CoarseBandEdgeFilter(UVodd,True)

    if TEFilter:
        UVeven.select(times = [UVeven.time_array[k*8128] for k in range(27)])
        UVodd.select(times = [UVodd.time_array[k*8128] for k in range(1,28)])

    #calculate the even minus the odd (second line is amplitude since vis. is complex)
    EMO = (UVeven.data_array-UVodd.data_array)
    EMOa = abs(np.array(EMO))
    SHAPE = EMOa.shape

    #calculate total number of data points
    N = np.prod(SHAPE)

    #turn into 1-D array for histogram
    EMOv = np.reshape(EMOa, N)

    #Get the flags into 1-D form from the UVData objects
    EMOvfeven = np.reshape(UVeven.flag_array, N)
    EMOvfodd = np.reshape(UVodd.flag_array, N)

    #Construct the conditionally filtered EMO arrays
    EMOvAND = [EMOv[k] for k in range(N) if EMOvfeven[k] and EMOvfodd[k]]
    EMOvNEITHER = [EMOv[k] for k in range(N) if not EMOvfeven[k] and not EMOvfodd[k]]
    EMOvXOR = [EMOv[k] for k in range(N) if EMOvfeven[k]^EMOvfodd[k]]

    return((EMOv, EMOvAND, EMOvNEITHER, EMOvXOR,SHAPE,UVeven.time_array,UVodd.time_array,UVeven.freq_array))
    

def EMOHist(filepath,OBSID,BEFilter,TEFilter):#('string',int,bool,bool)   

    HData = EMO(filepath, BEFilter, TEFilter)[0:4]
    MAX = max(HData[0])
    OPTIONS = [' [All Data]',' [No Band Edges, Yes Time Edges]',' [Yes Band Edges, No Time Edges]',' [No Band/Time Edges]']

    #selects indices to be used form OPTIONS list...
    def Opt(x,y):
        if not x and not y:
            return(OPTIONS[0])
        elif x and not y:
            return(OPTIONS[1])
        elif not x and y:
            return(OPTIONS[2])
        elif x and y:
            return(OPTIONS[3])

    #construct the histograms
    plt.hist(HData, bins = 1000, range = (0,MAX), histtype = 'step', label = ('ALL','AND','NEITHER','XOR'))
    plt.title('EMO Visibility Amplitude ObsID '+str(OBSID)+Opt(BEFilter,TEFilter))
    plt.yscale('log', nonposy = 'clip')
    plt.xlabel('|Veven-Vodd| (uncalib)')
    plt.ylabel('Counts')
    plt.xticks([0.1*MAX*k for k in range(11)])
    plt.ticklabel_format(axis = 'x', style = 'sci', scilimits = [-1,1])
    plt.legend()

    plt.show()

def NECompare(filepath1,filepath2, OBSID1, OBSID2, BEFilter,TEFilter):#('string','string',int,int,bool,bool)

    HData = (EMO(filepath1, BEFilter, TEFilter)[2], EMO(filepath2, BEFilter, TEFilter)[2])
    MAX = max([max(HData[0]),max(HData[1])])

    plt.hist(HData, bins = 1000, range = (0,MAX), histtype = 'step', label = ('Neither ' +str(OBSID1), 'Neither ' + str(OBSID2)))
    plt.title('EMO Vis. Amp. "Neither" Comparison, [no band/time edge]')
    plt.yscale('log', nonposy = 'clip')
    plt.xlabel('|Veven - Vodd| (uncalib)')
    plt.ylabel('Counts')
    plt.xticks([0.1*MAX*k for k in range(11)])
    plt.ticklabel_format(axis = 'x', style = 'sci', scilimits = [-1,1])
    plt.legend()

def TimeFreqHist(filepath, OBSID, BEFilter, TEFilter):

    emo = EMO(filepath, BEFilter, TEFilter)
    data = emo[0]
    shape = emo[4]
    Nfreqs = shape[2]
    Ntimes = shape[0]/8128
    nbins = 1000
    bins = np.linspace(0, max(data), nbins+1)
    ind = np.digitize(data, bins)

    F = np.reshape(ind, shape)
    G = np.zeros([Nfreqs,Ntimes])

    s = 0
    M = 0

    while bins[s] <= 0.25:
        M = s - 1
    
    for p in range(shape[0]):
        for q in range(Nfreqs):
             for r in range(4):
                 if F[p,0,q,r] >= M:
                     G[q,p/8128] += 1

    plt.imshow(G, cmap = cm.coolwarm, interpolation = 'none')
    plt.set_title('Time-Freq RFI Histogram ' + str(OBSID)+Opt(BEFilter,TEFilter))
    plt.xticks([0,5,10,15,20,25])
    plt.axes(aspect = 0.067)

    AVG = np.mean(G)
    MAX = np.amax(G)

    plt.colorbar(ticks = [0, AVG, MAX]).ax.set_yticklabels(['0', str(AVG), str(MAX)])

    plt.show()







    
