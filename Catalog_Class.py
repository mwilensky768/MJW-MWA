import plot_lib as pl
import rfiutil
import numpy as np

def RFI_Catalog(RFI, data_product=None, data_args=[], data_kwargs={}, plot_kwargs={}):

    """
    Data products include the INS, the one_d_histograms, waterfall histograms,
    filtered_INS, and bl_grid
    """

    data = getattr(RFI, data_product)(*data_args, **data_kwargs)

    plot_args = pl.plot_args(data_product)
    plots = getattr(pl, data_product)(*plot_args, **plot_kwargs)

    for i, prod in enumerate(data[:-1]):
        if type(prod) is np.ma.core.Masked_Array:
            prod.dump(data[-1][i])
        else:
            prod.save(data[-1][i])

    for i, fig in enumerate(plots[:-1]):
        fig.savefig(plots[-1][i])

class Catalog_Plot(type, N, plot_kwargs)
