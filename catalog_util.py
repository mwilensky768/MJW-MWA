def grid_setup(RFI):

    if RFI.UV.Npols == 4:
        gs = GridSpec(3, 2)
        gs_loc = [[1, 0], [1, 1], [2, 0], [2, 1]]
    elif RFI.UV.Npols == 2:
        gs = GridSpec(2, 2)
        gs_loc = [[1, 0], [1, 1]]
    else:
        gs = GridSpec(2, 1)
        gs_loc = [[1, 0], ]

    return(gs, gs_loc)
