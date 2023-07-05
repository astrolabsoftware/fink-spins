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
import io
import time
import requests

import pyspark.sql.functions as F
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import MapType, FloatType, StringType

from fink_utils.sso.utils import get_miriade_data
from fink_utils.sso.spins import estimate_sso_params

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from astropy.coordinates import SkyCoord
import astropy.units as u

import rocks

COLUMNS_ZTF = [
    'ssnamenr',
    'last_jd',
]

COLUMNS_FINK = [
    'H_1',
    'H_2',
    'G1_1',
    'G1_2',
    'G2_1',
    'G2_2',
    'R',
    'alpha0', # degree
    'delta0', # degree
    'obliquity', # degree
    'err_H_1',
    'err_H_2',
    'err_G1_1',
    'err_G1_2',
    'err_G2_1',
    'err_G2_2',
    'err_R',
    'err_alpha0', # degree
    'err_delta0', # degree
    'max_cos_lambda',
    'mean_cos_lambda',
    'min_cos_lambda',
    'min_phase', # degree
    'min_phase_1', # degree
    'min_phase_2', # degree
    'max_hase', # degree
    'max_phase_1', # degree
    'max_phase_2', # degree
    'chi2red',
    'rms',
    'rms_1',
    'rms_2',
    'mean_delta_RA_cos_DEC', # arcsecond
    'std_delta_RA_cos_DEC', # arcsecond
    'skew_delta_RA_cos_DEC', # arcsecond-2
    'kurtosis_delta_RA_cos_DEC', # arcsecond-3
    'mean_delta_DEC', # arcsecond
    'std_delta_DEC', # arcsecond
    'skew_delta_DEC', # arcsecond-2
    'kurtosis_delta_DEC', # arcsecond-3
    'n_obs',
    'n_obs_1',
    'n_obs_2',
    'n_days',
    'n_days_1',
    'n_days_2',
    'fit',
    'status',
    'flag',
    'version',
]

COLUMNS_SSODNET = [
    'sso_name',
    'sso_number',
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
            outdic['mean_delta_RA_cos_DEC'] = np.mean(deltaRAcosDEC)
            outdic['std_delta_RA_cos_DEC'] = np.std(deltaRAcosDEC)
            outdic['skew_delta_RA_cos_DEC'] = skew(deltaRAcosDEC)
            outdic['kurtosis_delta_RA_cos_DEC'] = kurtosis(deltaRAcosDEC)


            deltaDEC = (pdf['i:dec'] - pdf.Dec) * 3600
            outdic['mean_delta_DEC'] = np.mean(deltaDEC)
            outdic['std_delta_DEC'] = np.std(deltaDEC)
            outdic['skew_delta_DEC'] = skew(deltaDEC)
            outdic['kurtosis_delta_DEC'] = kurtosis(deltaDEC)

            # Time lapse
            outdic['n_days'] = len(pdf['i:jd'].values)
            ufilters = np.unique(pdf['i:fid'].values)
            for filt in ufilters:
                mask = pdf['i:fid'].values == filt
                outdic['n_days_{}'.format(filt)] = len(pdf['i:jd'].values[mask])

            outdic['last_jd'] = np.max(pdf['i:jd'].values)

            out.append(outdic)
    return pd.Series(out)

def rockify(ssnamenr: pd.Series):
    """ Extract names and numbers from ssnamenr

    Parameters
    ----------
    ssnamenr: pd.Series of str
        SSO names as given in ZTF alert packets

    Returns
    -----------
    sso_name: np.array of str
        SSO names according to quaero
    sso_number: np.array of int
        SSO numbers according to quaero
    """
    names_numbers = rocks.identify(ssnamenr)

    sso_name = np.transpose(names_numbers)[0]
    sso_number = np.transpose(names_numbers)[1]

    return sso_name, sso_number

def angular_separation(lon1, lat1, lon2, lat2):
    """
    Angular separation between two points on a sphere.
    Stolen from astropy -- for version <5

    Parameters
    ----------
    lon1, lat1, lon2, lat2 : `~astropy.coordinates.Angle`, `~astropy.units.Quantity` or float
        Longitude and latitude of the two points. Quantities should be in
        angular units; floats in radians.

    Returns
    -------
    angular separation : `~astropy.units.Quantity` ['angle'] or float
        Type depends on input; ``Quantity`` in angular units, or float in
        radians.

    Notes
    -----
    The angular separation is calculated using the Vincenty formula [1]_,
    which is slightly more complex and computationally expensive than
    some alternatives, but is stable at at all distances, including the
    poles and antipodes.

    .. [1] https://en.wikipedia.org/wiki/Great-circle_distance
    """
    sdlon = np.sin(lon2 - lon1)
    cdlon = np.cos(lon2 - lon1)
    slat1 = np.sin(lat1)
    slat2 = np.sin(lat2)
    clat1 = np.cos(lat1)
    clat2 = np.cos(lat2)

    num1 = clat2 * sdlon
    num2 = clat1 * slat2 - slat1 * clat2 * cdlon
    denominator = slat1 * slat2 + clat1 * clat2 * cdlon

    return np.arctan2(np.hypot(num1, num2), denominator)

def extract_obliquity(sso_name, alpha0, delta0, bft_file=None):
    """ Extract obliquity using spin values, and the BFT information

    Parameters
    ----------
    sso_name: np.array or pd.Series of str
        SSO names according to quaero (see `rockify`)
    alpha0: np.array or pd.Series of double
        RA of the pole [degree]
    delta0: np.array or pd.Series of double
        DEC of the pole [degree]
    bft_file: str, optional
        If given, read the BFT (parquet format). If not specified,
        download the latest version of the BFT (can be long).
        Default is None (unspecified).

    Returns
    ----------
    obliquity: np.array of double
        Obliquity for each object [degree]
    """
    if bft_file is None:
        print('BFT not found -- downloading...')
        r = requests.get('https://ssp.imcce.fr/data/ssoBFT-latest.parquet')
        pdf_bft = pd.read_parquet(io.BytesIO(r.content))
    else:
        pdf_bft = pd.read_parquet(bft_file)

    cols = ['sso_name', 'orbital_elements.node_longitude.value', 'orbital_elements.inclination.value']
    sub = pdf_bft[cols]

    pdf = pd.DataFrame(
        {
            'sso_name': sso_name,
            'alpha0': alpha0,
            'delta0': delta0
        }
    )

    pdf = pdf.merge(sub[cols], left_on='sso_name', right_on='sso_name', how='left' )

    # Orbit
    lon_orbit = (pdf['orbital_elements.node_longitude.value'] - 90).values
    lat_orbit = (90. - pdf['orbital_elements.inclination.value']).values

    # Spin -- convert to EC
    ra = np.nan_to_num(pdf.alpha0.values) * u.degree
    dec = np.nan_to_num(pdf.delta0.values) * u.degree

    # Trick to put the object "far enough"
    coords_spin = SkyCoord(ra=ra, dec=dec, distance=200*u.parsec, frame='hcrs')

    # in radian
    lon_spin = coords_spin.heliocentricmeanecliptic.lon.value
    lat_spin = coords_spin.heliocentricmeanecliptic.lat.value

    obliquity = np.degrees(
        angular_separation(
            lon_spin,
            lat_spin,
            np.radians(lon_orbit),
            np.radians(lat_orbit)
        )
    )

    return obliquity

def aggregate_sso_data(output_filename=None):
    """ Aggregate all SSO data in Fink

    Data is read from HDFS on VirtualData

    Parameters
    ----------
    output_filename: str, optional
        If given, save data on HDFS. Cannot overwrite. Default is None.

    Returns
    ----------
    df_grouped: Spark DataFrame
        Spark DataFrame with aggregated SSO data.
    """
    cols0 = ['candidate.ssnamenr']
    cols = [
        'candidate.ra',
        'candidate.dec',
        'candidate.magpsf',
        'candidate.sigmapsf',
        'candidate.fid',
        'candidate.jd'
    ]

    epochs = {
        'epoch1': [
            'archive/science/year=2019',
            'archive/science/year=2020',
            'archive/science/year=2021',
        ],
        'epoch2': [
            'archive/science/year=2022',
            'archive/science/year=2023',
        ]
    }

    df1 = spark.read.format('parquet').option('basePath', 'archive/science').load(epochs['epoch1'])
    df1.select(cols0 + cols)\
        .filter(F.col('ssnamenr') != 'null')\
        .groupBy('ssnamenr')\
        .agg(*[F.collect_list(col.split('.')[1]).alias('c' + col.split('.')[1]) for col in cols])

    df2 = spark.read.format('parquet').option('basePath', 'archive/science').load(epochs['epoch2'])
    df2.select(cols0 + cols)\
        .filter(F.col('ssnamenr') != 'null')\
        .groupBy('ssnamenr')\
        .agg(*[F.collect_list(col.split('.')[1]).alias('c' + col.split('.')[1]) for col in cols])


    df_union = df1.union(df2)

    df_grouped = df_union.groupBy('ssnamenr').agg(
        *[F.flatten(F.collect_list('c' + col.split('.')[1])).alias('c' + col.split('.')[1]) for col in cols]
    )

    if output_filename is not None:
        df_grouped.write.parquet('sso_aggregated')

    return df_grouped


def build_the_fft(aggregated_filename=None, nproc=80, nmin=50, model='SHG1G2') -> pd.DataFrame:
    """ Build the Fink Flat Table from scratch

    Parameters
    ----------
    aggregated_filename: str
        If given, read aggregated data on HDFS. Default is None.
    nproc: int, optional
        Number of cores to used. Default is 80.
    nmin: int, optional
        Minimal number of measurements to select objects (all filters). Default is 50.
    model: str, optional
        Model name among HG, HG1G2, SHG1G2. Default is SHG1G2.

    Returns
    ----------
    pdf: pd.DataFrame
        Pandas DataFrame with all the FFT data.
    """

    if aggregated_filename is not None:
        df_ztf = spark.read.format('parquet').load(aggregated_data)
    else:
        df_ztf = aggregate_sso_data()

    print('{:,} SSO objects in Fink'.format(df_ztf.count()))

    df = df_ztf\
        .withColumn('nmeasurements', F.size(df_ztf['cra']))\
        .filter(F.col('nmeasurements') >= nmin)\
        .repartition(nproc)\
        .cache()

    print('{:,} SSO objects with more than {} measurements'.format(df.count(), nmin))

    cols = ['ssnamenr', 'params']
    t0 = time.time()
    pdf = df\
        .withColumn(
            'params',
            estimate_sso_params_spark(
                F.col('ssnamenr').astype('string'),
                'cmagpsf',
                'csigmapsf',
                'cjd',
                'cfid',
                'cra',
                'cdec',
                F.lit('ephemcc'),
                F.lit(model)
            )
        ).select(cols).toPandas()

    print('Time to extract parameters: {:.2f}'.format(time.time() - t0))

    pdf = pd.concat([pdf, pd.json_normalize(pdf.params)], axis=1).drop('params', axis=1)

    sso_name, sso_number = rockify(pdf.ssnamenr)
    pdf['sso_name'] = sso_name
    pdf['sso_number'] = sso_number

    pdf['obliquity'] = extract_obliquity(pdf.sso_name, pdf.alpha0, pdf.delta0, bft_file=None)

    return pdf


