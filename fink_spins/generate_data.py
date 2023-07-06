# Copyright 2023 AstroLab Software
# Author: Roman Le Montagner, Benoit Carry
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pandas as pd
import requests
import io

import rocks
import sys


rocks.set_log_level("critical")


def format_HG(data_hg: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """
    Format the columns of the HG dataframe

    Parameters
    ----------
    data_hg : DataFrame
        HG data
    filters : dict
        filter dict

    Returns
    -------
    pd.DataFrame
        same as input but with columns rename and new columns to match the final dataframe of the script.

    Examples
    --------
    >>> HG_data = format_HG(data_hg, filters)
    >>> len(HG_data)
    115148
    >>> HG_data.columns
    Index(['ssnamenr', 'nmeasurements', 'ndays', 'params', 'number', 'name',
           'HG_H_g', 'HG_dH_g', 'HG_G_g', 'HG_dG_g', 'HG_rms_g', 'HG_H_r',
           'HG_dH_r', 'HG_G_r', 'HG_dG_r', 'HG_rms_r', 'minphase', 'maxphase',
           'n_days', 'n_obs', 'HG_rms', 'HG_chi2red', 'HG1G2_status', 'HG1G2_fit'],
          dtype='object')
    """

    # HG to data
    names_numbers = rocks.identify(data_hg.ssnamenr)
    data_hg["number"] = [nn[1] for nn in names_numbers]
    data_hg["name"] = [nn[0] for nn in names_numbers]

    for filt in filters.keys():
        data_hg["HG_H_{}".format(filters[filt])] = data_hg["params"].str[
            "H_{}".format(filt)
        ]
        data_hg["HG_dH_{}".format(filters[filt])] = data_hg["params"].str[
            "errH_{}".format(filt)
        ]
        data_hg["HG_G_{}".format(filters[filt])] = data_hg["params"].str[
            "G_{}".format(filt)
        ]
        data_hg["HG_dG_{}".format(filters[filt])] = data_hg["params"].str[
            "errG_{}".format(filt)
        ]
        data_hg["HG_rms_{}".format(filters[filt])] = data_hg["params"].str[
            "rms_{}".format(filt)
        ]

    data_hg["minphase"] = np.rad2deg(data_hg["params"].str["minphase"])
    data_hg["maxphase"] = np.rad2deg(data_hg["params"].str["maxphase"])

    data_hg["n_days"] = data_hg["ndays"]
    data_hg["n_obs"] = data_hg["nmeasurements"]

    data_hg["HG_rms"] = data_hg["params"].str["rms"]
    data_hg["HG_chi2red"] = data_hg["params"].str["chi2red"]

    data_hg["HG_status"] = data_hg["params"].str["status"]
    data_hg["HG_fit"] = data_hg["params"].str["fit"]

    cond = data_hg.name.notna()
    data_hg = data_hg[cond].reset_index(drop=True)

    return data_hg


def format_HG1G2(data_hg1g2: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """
    Format the columns of the HG1G2 dataframe

    Parameters
    ----------
    data_hg1g2 : DataFrame
        HG1G2 data
    filters : dict
        filter dict

    Returns
    -------
    pd.DataFrame
        same as input but with columns rename and new columns to match the final dataframe of the script.

    Examples
    --------
    >>> HG1G2_data = format_HG1G2(data_hg1g2, filters)
    >>> len(HG1G2_data)
    115238
    >>> HG1G2_data.columns
    Index(['ssnamenr', 'HG1G2_H_g', 'HG1G2_dH_g', 'HG1G2_G1_g', 'HG1G2_dG1_g',
           'HG1G2_G2_g', 'HG1G2_dG2_g', 'HG1G2_rms_g', 'HG1G2_H_r', 'HG1G2_dH_r',
           'HG1G2_G1_r', 'HG1G2_dG1_r', 'HG1G2_G2_r', 'HG1G2_dG2_r', 'HG1G2_rms_r',
           'HG1G2_rms', 'HG1G2_chi2red', 'HG1G2_status', 'HG1G2_fit'],
          dtype='object')
    """

    # HG1G2 to data
    for filt in filters.keys():
        data_hg1g2["HG1G2_H_{}".format(filters[filt])] = data_hg1g2["params"].str[
            "H_{}".format(filt)
        ]
        data_hg1g2["HG1G2_dH_{}".format(filters[filt])] = data_hg1g2["params"].str[
            "errH_{}".format(filt)
        ]
        data_hg1g2["HG1G2_G1_{}".format(filters[filt])] = data_hg1g2["params"].str[
            "G1_{}".format(filt)
        ]
        data_hg1g2["HG1G2_dG1_{}".format(filters[filt])] = data_hg1g2["params"].str[
            "errG1_{}".format(filt)
        ]
        data_hg1g2["HG1G2_G2_{}".format(filters[filt])] = data_hg1g2["params"].str[
            "G2_{}".format(filt)
        ]
        data_hg1g2["HG1G2_dG2_{}".format(filters[filt])] = data_hg1g2["params"].str[
            "errG2_{}".format(filt)
        ]
        data_hg1g2["HG1G2_rms_{}".format(filters[filt])] = data_hg1g2["params"].str[
            "rms_{}".format(filt)
        ]

    data_hg1g2["HG1G2_rms"] = data_hg1g2["params"].str["rms"]
    data_hg1g2["HG1G2_chi2red"] = data_hg1g2["params"].str["chi2red"]
    data_hg1g2["HG1G2_status"] = data_hg1g2["params"].str["status"]
    data_hg1g2["HG1G2_fit"] = data_hg1g2["params"].str["fit"]

    data_hg1g2 = data_hg1g2.drop(columns=["nmeasurements", "ndays", "params"])

    return data_hg1g2


def format_sHG1G2(data_shg1g2: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """
    Format the columns of the sHG1G2 dataframe

    Parameters
    ----------
    data_shg1g2 : DataFrame
        sHG1G2 data
    filters : dict
        filter dict

    Returns
    -------
    pd.DataFrame
        same as input but with columns rename and new columns to match the final dataframe of the script.

    Examples
    --------
    >>> sHG1G2_data = format_sHG1G2(data_shg1g2, filters)
    >>> len(sHG1G2_data)
    115238
    >>> sHG1G2_data.columns
    Index(['ssnamenr', 'sHG1G2_H_g', 'sHG1G2_dH_g', 'sHG1G2_G1_g', 'sHG1G2_dG1_g',
           'sHG1G2_G2_g', 'sHG1G2_dG2_g', 'sHG1G2_rms_g', 'sHG1G2_H_r',
           'sHG1G2_dH_r', 'sHG1G2_G1_r', 'sHG1G2_dG1_r', 'sHG1G2_G2_r',
           'sHG1G2_dG2_r', 'sHG1G2_rms_r', 'sHG1G2_RA0', 'sHG1G2_dRA0',
           'sHG1G2_DEC0', 'sHG1G2_dDEC0', 'sHG1G2_R', 'sHG1G2_dR', 'sHG1G2_rms',
           'sHG1G2_chi2red', 'sHG1G2_status', 'sHG1G2_fit', 'sHG1G2_minCosLambda',
           'sHG1G2_meanCosLambda', 'sHG1G2_maxCosLambda'],
          dtype='object')
    """

    # HG1G2hyb to data
    for filt in filters.keys():
        data_shg1g2["sHG1G2_H_{}".format(filters[filt])] = data_shg1g2["params"].str[
            "H_{}".format(filt)
        ]
        data_shg1g2["sHG1G2_dH_{}".format(filters[filt])] = data_shg1g2["params"].str[
            "errH_{}".format(filt)
        ]
        data_shg1g2["sHG1G2_G1_{}".format(filters[filt])] = data_shg1g2["params"].str[
            "G1_{}".format(filt)
        ]
        data_shg1g2["sHG1G2_dG1_{}".format(filters[filt])] = data_shg1g2["params"].str[
            "errG1_{}".format(filt)
        ]
        data_shg1g2["sHG1G2_G2_{}".format(filters[filt])] = data_shg1g2["params"].str[
            "G2_{}".format(filt)
        ]
        data_shg1g2["sHG1G2_dG2_{}".format(filters[filt])] = data_shg1g2["params"].str[
            "errG2_{}".format(filt)
        ]
        data_shg1g2["sHG1G2_rms_{}".format(filters[filt])] = data_shg1g2["params"].str[
            "rms_{}".format(filt)
        ]

    data_shg1g2["sHG1G2_RA0"] = np.degrees(data_shg1g2["params"].str["alpha0"])
    data_shg1g2["sHG1G2_dRA0"] = np.degrees(data_shg1g2["params"].str["erralpha0"])
    data_shg1g2["sHG1G2_DEC0"] = np.degrees(data_shg1g2["params"].str["delta0"])
    data_shg1g2["sHG1G2_dDEC0"] = np.degrees(data_shg1g2["params"].str["errdelta0"])
    data_shg1g2["sHG1G2_R"] = data_shg1g2["params"].str["R"]
    data_shg1g2["sHG1G2_dR"] = data_shg1g2["params"].str["errR"]

    data_shg1g2["sHG1G2_rms"] = data_shg1g2["params"].str["rms"]
    data_shg1g2["sHG1G2_chi2red"] = data_shg1g2["params"].str["chi2red"]
    data_shg1g2["sHG1G2_status"] = data_shg1g2["params"].str["status"]
    data_shg1g2["sHG1G2_fit"] = data_shg1g2["params"].str["fit"]
    data_shg1g2["sHG1G2_minCosLambda"] = data_shg1g2["params"].str["minCosLambda"]
    data_shg1g2["sHG1G2_meanCosLambda"] = data_shg1g2["params"].str["meanCosLambda"]
    data_shg1g2["sHG1G2_maxCosLambda"] = data_shg1g2["params"].str["maxCosLambda"]

    data_shg1g2 = data_shg1g2.drop(columns=["nmeasurements", "ndays", "params"])

    return data_shg1g2


def format_ssoft(data_shg1g2: pd.DataFrame, filters: dict) -> pd.DataFrame:
    dict_rename = {
        col: f"sHG1G2_{col}" for col in data_shg1g2.columns if col != "ssnamenr"
    }
    for filt in filters.keys():
        dict_rename[f"H_{filt}"] = f"sHG1G2_H_{filters[filt]}"
        dict_rename[f"err_H_{filt}"] = f"sHG1G2_dH_{filters[filt]}"
        dict_rename[f"G1_{filt}"] = f"sHG1G2_G1_{filters[filt]}"
        dict_rename[f"G2_{filt}"] = f"sHG1G2_G2_{filters[filt]}"
        dict_rename[f"err_G1_{filt}"] = f"sHG1G2_dG1_{filters[filt]}"
        dict_rename[f"err_G2_{filt}"] = f"sHG1G2_dG2_{filters[filt]}"
        dict_rename[f"rms_{filt}"] = f"sHG1G2_rms_{filters[filt]}"
        dict_rename[f"min_phase_{filt}"] = f"sHG1G2_min_phase_{filters[filt]}"
        dict_rename[f"max_phase_{filt}"] = f"sHG1G2_max_phase_{filters[filt]}"
        dict_rename[f"n_obs_{filt}"] = f"sHG1G2_n_obs_{filters[filt]}"
        dict_rename[f"n_days_{filt}"] = f"sHG1G2_n_days_{filters[filt]}"

    dict_rename["alpha0"] = "sHG1G2_RA0"
    dict_rename["delta0"] = "sHG1G2_DEC0"
    dict_rename["err_alpha0"] = "sHG1G2_dRA0"
    dict_rename["err_delta0"] = "sHG1G2_dDEC0"

    return data_shg1g2.rename(dict_rename, axis="columns")


if __name__ == "__main__":
    # Local Configuration
    data_fink = "data/ztf/"

    # ZTF filters 1: g, 2: r
    filters = {"1": "g", "2": "r"}

    r = requests.post(
        sys.argv[1],
        json={"output-format": "parquet"},
    )
    data_shg1g2 = pd.read_parquet(io.BytesIO(r.content))
    # data_shg1g2 = pd.read_parquet(f"{data_fink}sso_fink_SHG1G2_sign.parquet")

    # data_shg1g2 = pd.read_parquet(f"{data_fink}sso_fink_SHG1G2.parquet")  # All
    data_hg1g2 = pd.read_parquet(f"{data_fink}sso_fink_HG1G2.parquet")
    data_hg = pd.read_parquet(f"{data_fink}sso_fink_HG.parquet")

    print(
        "check data:\n\tlen(HG): {}\n\tlen(HG1G2): {}\n\tlen(sHG1G2): {}".format(
            len(data_hg), len(data_hg1g2), len(data_shg1g2)
        )
    )

    data_hg = format_HG(data_hg, filters)
    data_hg1g2 = format_HG1G2(data_hg1g2, filters)
    data_shg1g2 = format_ssoft(data_shg1g2, filters)

    # HG with HG1G2
    data_2 = data_hg.merge(data_hg1g2, on="ssnamenr")

    # (HG with HG1G2) with sHG1G2
    data = data_2.merge(data_shg1g2, on="ssnamenr")

    print(
        "check data:\n\tlen(HG): {}\n\tlen(HG1G2): {}\n\tlen(sHG1G2): {}\n\tlen(final data format): {}".format(
            len(data_hg), len(data_hg1g2), len(data_shg1g2), len(data)
        )
    )

    data = data.astype({"number": "Int64"})

    data.to_parquet(f"{data_fink}sso_ZTF.parquet")
