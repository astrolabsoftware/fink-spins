# Copyright 2023 AstroLab Software
# Author: Julien Peloton
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
""" This file contains scripts and definition for the Fink BFT
"""
import pyspark.sql.functions as F
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import MapType, FloatType, StringType

from fink_utils.sso.utils import get_miriade_data
from fink_utils.sso.spins import estimate_sso_params

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

MODELS = {
    'HG': {
        'p0': [15.0, 0.15],
        'bounds': ([0, 0], [30, 1])
    },
    'HG1G2': {
        'p0': [15.0, 0.15, 0.15],
        'bounds': ([0, 0, 0], [30, 1, 1])
    },
    'SHG1G2': {
        'p0': [15.0, 0.15, 0.15, 0.8, np.pi, 0.0],
        'bounds': ([0, 0, 0, 1e-1, 0, -np.pi / 2], [30, 1, 1, 1, 2 * np.pi, np.pi / 2])
    },
}

COLUMNS_ZTF = [
    'ssnamenr',
    'jd', # last update?
]

COLUMNS_FINK = [
    'H_1',
    'H_2',
    'G1_1',
    'G1_2',
    'G2_1',
    'G2_2',
    'R',
    'alpha0',
    'delta0',
    'obliquity', # a posteriori
    'errH_1',
    'errH_2',
    'errG1_1',
    'errG1_2',
    'errG2_1',
    'errG2_2',
    'errR',
    'erralpha0',
    'errdelta0',
    'maxCosLambda',
    'meanCosLambda',
    'minCosLambda',
    'minphase',
    'minphase_1',
    'minphase_2',
    'maxphase',
    'maxphase_1',
    'maxphase_2',
    'chi2red',
    'rms',
    'rms_1',
    'rms_2',
    'meanDeltaRAcosDEC',
    'stdDeltaRAcosDEC',
    'skewDeltaRAcosDEC',
    'kurtosisDeltaRAcosDEC',
    'meanDeltaDEC',
    'stdDeltaDEC',
    'skewDeltaDEC',
    'kurtosisDeltaDEC',
    'nobs',
    'nobs_1',
    'nobs_2',
    'ndays',
    'ndays_1',
    'ndays_2',
    'fit',
    'status'
]

COLUMNS_SSODNET = [
    'name', # a posteriori
    'number', # a posteriori
]

@pandas_udf(MapType(StringType(), FloatType()), PandasUDFType.SCALAR)
def estimate_sso_params_spark(ssnamenr, magpsf, sigmapsf, jd, fid, ra, dec, method, model):
    """ Extract phase and spin parameters from Fink alert data using Apache Spark

    Parameters
    ----------
    ssnamenr: str
        SSO name from ZTF alert packet
    magpsf: float
        Magnitude from ZTF
    sigmapsf: float
        Error estimate on magnitude
    jd: double
        Time of exposition (UTC)
    fid: int
        Filter ID (1=g, 2=r)
    ra: double
        RA
    dec: double
        Declination
    method: str
        Method to compute ephemerides: `ephemcc` or `rest`.
        Use only the former on the Spark Cluster (local installation of ephemcc),
        otherwise use `rest` to call the ssodnet web service.
    model: str
        Model name. Available: HG, HG1G2, SHG1G2

    Returns
    ----------
    out: pd.Series
        Series with dictionaries. Keys are parameter names (H, G, etc.)
        depending on the model chosen.
    """
    # loop over SSO
    out = []
    for index, ssname in enumerate(ssnamenr.values):

        # First get ephemerides data
        pdf_sso = pd.DataFrame(
            {
                'i:ssnamenr': [ssname] * len(ra.values[index]),
                'i:magpsf': magpsf.values[index],
                'i:sigmapsf': sigmapsf.values[index],
                'i:jd': jd.values[index],
                'i:fid': fid.values[index],
                'i:ra': ra.values[index],
                'i:dec': dec.values[index]
            }
        )

        pdf_sso = pdf_sso.sort_values('i:jd')

        if method.values[0] == 'ephemcc':
            # hardcoded for the Spark cluster!
            parameters = {
                'outdir': '/tmp/ramdisk/spins',
                'runner_path': '/tmp/fink_run_ephemcc4.sh',
                'userconf': '/tmp/.eproc-4.3',
                'iofile': '/tmp/default-ephemcc-observation.xml'
            }
        elif method.values[0] == 'rest':
            parameters = {}
        pdf = get_miriade_data(
            pdf_sso,
            observer='I41',
            rplane='1',
            tcoor=5,
            withecl=False,
            method=method.values[0],
            parameters=parameters
        )

        if 'i:magpsf_red' not in pdf.columns:
            out.append({'fit': 2, 'status': -2})
        else:

            if model.values[0] == 'SHG1G2':
                outdic = estimate_sso_params(
                    pdf['i:magpsf_red'].values,
                    pdf['i:sigmapsf'].values,
                    np.deg2rad(pdf['Phase'].values),
                    pdf['i:fid'].values,
                    np.deg2rad(pdf['i:ra'].values),
                    np.deg2rad(pdf['i:dec'].values),
                    p0=MODELS[model.values[0]]['p0'],
                    bounds=MODELS[model.values[0]]['bounds'],
                    model=model.values[0],
                    normalise_to_V=False
                )
            else:
                outdic = estimate_sso_params(
                    pdf['i:magpsf_red'].values,
                    pdf['i:sigmapsf'].values,
                    np.deg2rad(pdf['Phase'].values),
                    pdf['i:fid'].values,
                    p0=MODELS[model.values[0]]['p0'],
                    bounds=MODELS[model.values[0]]['bounds'],
                    model=model.values[0],
                    normalise_to_V=False
                )

            # Add astrometry
            deltaRAcosDEC = (pdf['i:ra'] - pdf.RA) * np.cos(np.radians(pdf['i:dec'])) * 3600
            outdic['meanDeltaRAcosDEC'] = np.mean(deltaRAcosDEC)
            outdic['stdDeltaRAcosDEC'] = np.std(deltaRAcosDEC)
            outdic['skewDeltaRAcosDEC'] = skew(deltaRAcosDEC)
            outdic['kurtosisDeltaRAcosDEC'] = kurtosis(deltaRAcosDEC)


            deltaDEC = (pdf['i:dec'] - pdf.Dec) * 3600
            outdic['meanDeltaDEC'] = np.mean(deltaDEC)
            outdic['stdDeltaDEC'] = np.std(deltaDEC)
            outdic['skewDeltaDEC'] = skew(DeltaDEC)
            outdic['kurtosisDeltaDEC'] = kurtosis(DeltaDEC)

            # Time lapse
            outdic['ndays'] = len(pdf['jd'].values)
            ufilters = np.unique(fid)
            for filt in ufilters:
                mask = fid == filt
                outdic['ndays_{}'.format(filt)] = len(pdf['jd'].values[mask])

            out.append(outdic)
    return pd.Series(out)