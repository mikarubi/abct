from typing import Tuple

import os
import numpy as np
import nibabel as nib

def get_plot_data():

    ## Get surface data
    n_hemi = 32492
    surf_name_L = "S1200.L.very_inflated_MSMAll.32k_fs_LR.surf.gii"
    surf_name_R = "S1200.R.very_inflated_MSMAll.32k_fs_LR.surf.gii"
    surf_file_L = os.path.join(".", "data", surf_name_L)
    surf_file_R = os.path.join(".", "data", surf_name_R)
    parc_name = "Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii"
    parc_file = os.path.join(".", "data", parc_name)
    gii_left = [arr.data for arr in nib.load(surf_file_L).darrays]
    gii_right = [arr.data for arr in nib.load(surf_file_R).darrays]
    cii = np.array(nib.load(parc_file).get_fdata())

    return gii_left, gii_right, cii, n_hemi








