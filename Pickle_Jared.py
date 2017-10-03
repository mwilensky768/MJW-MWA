import numpy as np
import matplotlib.pyplot as plt
import pickle
def NaNaNaNaNaNaNaNa():
    xs = np.arange(-7.25, 7.25, 0.01)
    ys = np.arange(-5, 5, 0.01)
    x, y = np.meshgrid(xs, ys)
    eq1 = ((x/7)**2*np.sqrt(abs(abs(x)-3)/(abs(x)-3))+(y/3)**2*np.sqrt(abs(y+3/7*np.sqrt(33))/(y+3/7*np.sqrt(33)))-1)
    eq2 = (abs(x/2)-((3*np.sqrt(33)-7)/112)*x**2-3+np.sqrt(1-(abs(abs(x)-2)-1)**2)-y)
    eq3 = (9*np.sqrt(abs((abs(x)-1)*(abs(x)-.75))/((1-abs(x))*(abs(x)-.75)))-8*abs(x)-y)
    eq4 = (3*abs(x)+.75*np.sqrt(abs((abs(x)-.75)*(abs(x)-.5))/((.75-abs(x))*(abs(x)-.5)))-y)
    eq5 = (2.25*np.sqrt(abs((x-.5)*(x+.5))/((.5-x)*(.5+x)))-y)
    eq6 = (6*np.sqrt(10)/7+(1.5-.5*abs(x))*np.sqrt(abs(abs(x)-1)/(abs(x)-1))-(6*np.sqrt(10)/14)*np.sqrt(4-(abs(x)-1)**2)-y)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for f in [eq1,eq2,eq3,eq4,eq5,eq6]:
        ax.contour(x, y, f, [0])

    pickle.dump(ax, open('foo.p', 'wb'))
    pickle.dump(fig, open('bar.p', 'wb'))
    #return fig, ax
### END def NaNaNaNaNaNaNaNa
if __name__ == '__main__':
    #fig, ax = NaNaNaNaNaNaNaNa()
    #NaNaNaNaNaNaNaNa()
    ax_reloaded = pickle.load(open('foo.p', 'rb'))
    #fig_reloaded = pickle.load(open('bar.p', 'rb'))
    ax_reloaded.text(-6, 4, 'The Axes object survives pickling!')
    plt.show()
