# Copyright 2023 AstroLab Software
# Author: Roman Le Montagner, Benoit Carry, Julien Peloton
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
""" Script to generate aggregated data from the SSOFT """
import requests
import argparse
import logging
import io
import os

import pandas as pd

import rocks

rocks.set_log_level("critical")


def format_ssoft(data: pd.DataFrame, filters: dict, flavor: str) -> pd.DataFrame:
    """ Formatting of the SSOFT columns
    """
    shared_cols = [
        'min_phase',
        'min_phase_1',
        'min_phase_2',
        'max_phase',
        'max_phase_1',
        'max_phase_2',
        'n_obs',
        'n_obs_1',
        'n_obs_2',
        'n_days',
        'n_days_1',
        'n_days_2',
        'mean_astrometry',
        'std_astrometry',
        'skew_astrometry',
        'kurt_astrometry',
        'last_jd',
        'version',
        'sso_name',
        'sso_number'
    ]
    if flavor in ['HG1G2', 'HG']:
        # Remove columns that are shared across all SSOFT
        data = data.drop(columns=shared_cols, errors='ignore')

    dict_rename = {
        col: f"{flavor}_{col}" for col in data.columns if col not in (shared_cols + ['ssnamenr'])
    }
    for filt in filters.keys():
        dict_rename[f"H_{filt}"] = f"{flavor}_H_{filters[filt]}"
        dict_rename[f"err_H_{filt}"] = f"{flavor}_dH_{filters[filt]}"
        dict_rename[f"rms_{filt}"] = f"{flavor}_rms_{filters[filt]}"
        dict_rename[f"min_phase_{filt}"] = f"min_phase_{filters[filt]}"
        dict_rename[f"max_phase_{filt}"] = f"max_phase_{filters[filt]}"
        dict_rename[f"n_obs_{filt}"] = f"n_obs_{filters[filt]}"
        dict_rename[f"n_days_{filt}"] = f"n_days_{filters[filt]}"

    if flavor == 'SHG1G2':
        for filt in filters.keys():
            dict_rename[f"G1_{filt}"] = f"{flavor}_G1_{filters[filt]}"
            dict_rename[f"G2_{filt}"] = f"{flavor}_G2_{filters[filt]}"
            dict_rename[f"err_G1_{filt}"] = f"{flavor}_dG1_{filters[filt]}"
            dict_rename[f"err_G2_{filt}"] = f"{flavor}_dG2_{filters[filt]}"

        dict_rename["alpha0"] = f"{flavor}_alpha0"
        dict_rename["delta0"] = f"{flavor}_delta0"
        dict_rename["err_alpha0"] = f"{flavor}_dalpha0"
        dict_rename["err_delta0"] = f"{flavor}_ddelta0"
        dict_rename["err_R"] = f"{flavor}_dR"
    elif flavor == 'HG1G2':
        for filt in filters.keys():
            dict_rename[f"G1_{filt}"] = f"{flavor}_G1_{filters[filt]}"
            dict_rename[f"G2_{filt}"] = f"{flavor}_G2_{filters[filt]}"
            dict_rename[f"err_G1_{filt}"] = f"{flavor}_dG1_{filters[filt]}"
            dict_rename[f"err_G2_{filt}"] = f"{flavor}_dG2_{filters[filt]}"
    elif flavor == 'HG':
        for filt in filters.keys():
            dict_rename[f"G_{filt}"] = f"{flavor}_G_{filters[filt]}"
            dict_rename[f"err_G_{filt}"] = f"{flavor}_dG_{filters[filt]}"

    return data.rename(dict_rename, axis="columns")


if __name__ == "__main__":
    """
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '-version', type=str, default='2023.08',
        help="Version of the SSOFT. Default is 2023.08")
    parser.add_argument(
        '-output_dir', type=str, default='data/ztf/',
        help="Output folder to store data. Default is data/ztf")
    parser.add_argument(
        '-api_url', type=str, default='https://fink-portal.org',
        help="API URL from Fink. Default is https://fink-portal.org.")
    parser.add_argument(
        '--rocks', action='store_true',
        help="If specified, override names & numbers from the SSOFT by running rocks on `ssnamenr`.")
    parser.add_argument(
        '--verbose', action='store_true',
        help="If specified, print on screen information about the processing.")
    args = parser.parse_args(None)

    os.makedirs(args.output_dir, exist_ok=True)
    # ZTF filters 1: g, 2: r
    filters = {"1": "g", "2": "r"}

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.info('Downloading the SSOFT from Fink servers...')

    flavors = ['SHG1G2', 'HG1G2', 'HG']
    container = []
    for flavor in flavors:
        r = requests.post(
            '{}/api/v1/ssoft'.format(args.api_url),
            json={
                "version": args.version,
                "flavor": flavor,
                "output-format": "parquet"
            },
        )
        container.append(pd.read_parquet(io.BytesIO(r.content)))

    if args.verbose:
        msg = "check data:\n\tlen({}): {}\n\tlen({}): {}\n\tlen({}): {}".format(
            *[y for x in zip(flavors, [len(i) for i in container]) for y in x]
        )
        logging.info(msg)

    if args.verbose:
        logging.info('Merging the 3 SSOFTs...')

    for index in range(len(container)):
        # inplace replacement
        container[index] = format_ssoft(container[index], filters, flavor=flavors[index])

    # HG with HG1G2
    tmp = container[2].merge(container[1], on="ssnamenr")
    # (HG with HG1G2) with sHG1G2
    data = tmp.merge(container[0], on="ssnamenr")

    if args.rocks:
        logging.info('Identifying names with rocks...')
        names_numbers = rocks.identify(data.ssnamenr)
        data["number"] = [nn[1] for nn in names_numbers]
        data["name"] = [nn[0] for nn in names_numbers]
    else:
        # Names are already given in the SSOFT
        dict_rename = {'sso_number': 'number', 'sso_name': 'name'}
        data = data.rename(dict_rename, axis="columns")

    data = data.astype({"number": "Int64"})

    if args.verbose:
        logging.info('Here is the first row ({} columns): '.format(len(data.columns)))
        logging.info(data.head(1).to_dict(orient='index'))

    if args.verbose:
        logging.info('Saving to {}'.format(args.output_dir))
    data.to_parquet("{}/sso_ZTF.parquet".format(args.output_dir))
