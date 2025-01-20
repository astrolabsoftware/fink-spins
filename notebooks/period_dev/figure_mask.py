""" This file contains the definition of all masks used in figures
"""
import os
import pandas as pd
import numpy as np

def compute_mask(data, R_min=0.3, thres_spin=1e-3, thres_phase=1e-3, model='FINK', kind='inter'):
    """Extract Fink mask
    
    Parameters
    ----------
    data: Pandas DataFrame
        Data containing the processed SSOFT (all models)
    R_min: float, optional
        Minimum oblateness. Default is 0.3
    thres: float, optional
        Threshold for selection of non-zero spin values, in degree. 
        Default is 1e-3.
    model: str, optional
        Model to choose:
            - FINK (fit + phase + R + spin)
            - 
    kind: str, optional
        Choose the intersection or the union of bands for phase parameters
        
        
    Returns
    ----------
    maskFINK: Pandas Series
        Series containing boolean
    """
    if model == 'FINK':
        mask_R = compute_mask_R(data, R_min)
        mask_phase_shg1g2 = compute_mask_phase(data, thres_phase, kind, model='SHG1G2')
        mask_phase_sshg1g2 = compute_mask_phase(data, thres_phase, kind, model='SSHG1G2')
        maskSpin_shg1g2 = compute_mask_spin(data, thres_spin, model="SHG1G2")
        maskSpin_sshg1g2 = compute_mask_spin(data, thres_spin, model="SSHG1G2")
        mask = mask_phase_shg1g2 & mask_phase_sshg1g2 & mask_R & maskSpin_shg1g2 & maskSpin_sshg1g2
    else:
        mask = compute_mask_phase(data, thres_phase, kind, model=model)
    
    return mask


def compute_sg1g2_mask(data, R_min=0.3, thres=1e-3, kind='inter'):
    """Extract Fink mask
    
    Parameters
    ----------
    data: Pandas DataFrame
        Data containing the processed SSOFT (all models)
    R_min: float, optional
        Minimum oblateness. Default is 0.3
    thres: float, optional
        Threshold for selection of non-zero spin values, in degree. 
        Default is 1e-3.
    kind: str, optional
        Choose the intersection or the union of bands for phase parameters
        
        
    Returns
    ----------
    maskFINK: Pandas Series
        Series containing boolean
    """
    
    mask_phase = compute_mask_phase(data, thres, kind, model='HG1G2')
    
    return mask

def compute_mask_R(data, R_min):
    """ Compute mask for good oblateness values
    """
    mask_R = (data.SHG1G2_R > R_min)
    
    return mask_R

def compute_mask_spin(data, thres, model="SHG1G2"):
    """ Compute mask for good spin values
    """
    if model == "SSHG1G2":
        maskSpin = (
            (data.SSHG1G2_alpha0.notna()) 
            & (data.SSHG1G2_delta0.notna()) 
            & (data.SSHG1G2_alpha0 > thres)
            & (np.abs(360 - data.SSHG1G2_alpha0) > thres)
            & (np.abs(data.SSHG1G2_alpha0 - 180) > thres)
            & (np.abs(data.SSHG1G2_delta0) > thres)
        )
    elif model == "SHG1G2":
        maskSpin = (
            (data.SHG1G2_alpha0.notna()) 
            & (data.SHG1G2_delta0.notna()) 
            & (data.SHG1G2_alpha0 > thres)
            & (np.abs(360 - data.SHG1G2_alpha0) > thres)
            & (np.abs(data.SHG1G2_alpha0 - 180) > thres)
            & (np.abs(data.SHG1G2_delta0) > thres)
        )
    return maskSpin

def compute_mask_phase(data, thres, kind, model='SHG1G2'):
    """ Compute mask for good phase values
    """
    if model == 'SSHG1G2':
        mask_fit = (data.SSHG1G2_fit == 0) & (data.SSHG1G2_status >= 2)
        mask_g = (
            (data.SSHG1G2_G1_g > thres)
            & (data.SSHG1G2_G2_g > thres)
            & ((1 - data.SSHG1G2_G1_g - data.SSHG1G2_G2_g) > thres)
        )
        mask_r = (
            (data.SSHG1G2_G1_r > thres)
            & (data.SSHG1G2_G2_r > thres)
            & ((1 - data.SSHG1G2_G1_r - data.SSHG1G2_G2_r) > thres)
        )
    elif model == 'SHG1G2':
        mask_fit = (data.SHG1G2_fit == 0) & (data.SHG1G2_status >= 2)
        mask_g = (
            (data.SHG1G2_G1_g > thres)
            & (data.SHG1G2_G2_g > thres)
            & ((1 - data.SHG1G2_G1_g - data.SHG1G2_G2_g) > thres)
        )
        mask_r = (
            (data.SHG1G2_G1_r > thres)
            & (data.SHG1G2_G2_r > thres)
            & ((1 - data.SHG1G2_G1_r - data.SHG1G2_G2_r) > thres)
        )
    elif model == 'HG1G2':
        mask_fit = (data.HG1G2_fit == 0) & (data.HG1G2_status >= 2)
        mask_g = (
            (data.HG1G2_G1_g > thres)
            & (data.HG1G2_G2_g > thres)
            & ((1 - data.HG1G2_G1_g - data.HG1G2_G2_g) > thres)
        )
        mask_r = (
            (data.HG1G2_G1_r > thres)
            & (data.HG1G2_G2_r > thres)
            & ((1 - data.HG1G2_G1_r - data.HG1G2_G2_r) > thres)
        )
    elif model == 'HG':
        mask_fit = (data.HG_fit == 0) & (data.HG_status >= 2)
        mask_g = (data.HG_H_g.notna()) & (data.HG_G_g.notna())
        mask_r = (data.HG_H_r.notna()) & (data.HG_G_r.notna())
    
    if kind == 'inter':
        mask = mask_fit & (mask_g & mask_r)
    elif kind == 'union':
        mask = mask_fit & (mask_g | mask_r)

    return mask
    
def print_statistics(data, model='FINK'):
    """
    """
    mask_inter = compute_mask(data, model=model, kind='inter')
    mask_union = compute_mask(data, model=model, kind='union')
    
    print(
        f"  All data       : {len(data):6d}  ({100:>6.2f}%)")
    print(
        f"  Mask {model} (g∩r) : {len(data[mask_inter]):6d}  ({100.*len(data[mask_inter])/len(data):>6.2f}%)"
    )
    print(
        f"  Mask {model} (gUr) : {len(data[mask_union]):6d}  ({100.*len(data[mask_union])/len(data):>6.2f}%)"
    )
        

if __name__ == "__main__":
    data_fink = '../'
    data = pd.read_parquet(os.path.join(data_fink, 'data', 'ztf', 'sso_ZTF.parquet'))
    
    for model in ['HG', 'HG1G2', 'SHG1G2', 'FINK']:
        print_statistics(data, model=model)
        print()
    
    