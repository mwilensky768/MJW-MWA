import catalog_plot as cp
import rfiutil
import numpy as np
import os
import glob


def Catalog_Generate(RFI, data_product, data_args=(), data_kwargs={},
                     plot_kwargs={}):

    data = getattr(RFI, data_product)(*data_args, **data_kwargs)
    fig_outpath = '%s/figs/%s/' % (RFI.outpath, data_product)
    if not os.path.exists(fig_outpath):
        os.makedirs(fig_outpath)
    getattr(cp, data_product)(RFI, data, fig_outpath, data_kwargs=data_kwargs,
                              plot_kwargs=plot_kwargs)
