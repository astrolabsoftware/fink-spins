import numpy as np
import pandas as pd

import requests
import io

import astropy.units as u
from astropy.coordinates import SkyCoord
import notebooks.ssptools
import fink_utils.sso.spins as finkus
from abc import ABC, abstractmethod


class Model(ABC):
    # Local Configuration
    data_fink = "../"
    # ZTF Colors and Colors
    colors = ["#15284F", "#F5622E"]
    dict_band = {"g": 1, "r": 2}

    def __init__(self, band: str, target: str = None) -> None:
        self.data = pd.read_parquet(f"{Model.data_fink}data/ztf/sso_ZTF.parquet")
        self.target = target
        self.reset_filter(band)
        if target is not None:
            self.reset_target(self.target)

    def reset_target(self, target):
        if target is None:
            raise Exception(
                f"target is None, use 'reset_target' to set a target for this model {self}"
            )
        # Get ZTF observations
        r = requests.post(
            "https://fink-portal.org/api/v1/sso",
            json={"n_or_d": target, "withEphem": True, "output-format": "json"},
        )

        # Format output in a DataFrame
        self.ztf = pd.read_json(io.BytesIO(r.content))

        # Compute Ephemerides
        jd_min = self.ztf["Date"].min()
        jd_max = self.ztf["Date"].max()
        self.step = 2
        nbd = (jd_max - jd_min) / self.step

        self.eph = notebooks.ssptools.ephemcc(
            target, ep=jd_min, nbd=nbd, step=f"{self.step}d"
        )

    def reset_filter(self, band):
        self.filter = band
        self.int_filter = Model.dict_band[self.filter]

    # From apparent magnitude to unit distance
    def dist_reduction(d_obs, d_sun):
        return 5 * np.log10(d_obs * d_sun)

    @abstractmethod
    def mag_model(self):
        pass

    def mag_dist(self, x_model, x_red):
        return self.mag_model(x_model) + Model.dist_reduction(*x_red)

    def mag_dist_eph(self):
        return self.mag_dist(
            np.radians(self.eph["Phase"]), (self.eph["Dobs"], self.eph["Dhelio"])
        )

    def mag_dist_ztf(self):
        cond = self.ztf["i:fid"] == self.int_filter
        return self.mag_dist(
            np.radians(self.ztf.loc[cond, "Phase"]),
            (self.ztf.loc[cond, "Dobs"], self.ztf.loc[cond, "Dhelio"]),
        )

    def residuals(self):
        cond = self.ztf["i:fid"] == self.int_filter
        return self.ztf.loc[cond, "i:magpsf"] - self.mag_dist_ztf()

    def get_chi2red(self):
        return self.data.loc[
            self.data.ssnamenr == self.target, f"{self.model_name}_chi2red"
        ].values[0]


class HG(Model):
    def __init__(self, band: str, target: str = None) -> None:
        super().__init__(band, target)
        self.model_name = "HG"

    def get_data_model(self):
        H = self.data.loc[
            self.data.ssnamenr == self.target, f"{self.model_name}_H_{self.filter}"
        ].values[0]
        G = self.data.loc[
            self.data.ssnamenr == self.target, f"{self.model_name}_G_{self.filter}"
        ].values[0]
        return H, G

    def mag_model(self, x):
        H, G = self.get_data_model()
        return finkus.func_hg(x, H, G)


class HG1G2(Model):
    def __init__(self, band: str, target: str = None) -> None:
        super().__init__(band, target)
        self.model_name = "HG1G2"

    def get_data_model(self):
        H = self.data.loc[
            self.data.ssnamenr == self.target, f"{self.model_name}_H_{self.filter}"
        ].values[0]
        G1 = self.data.loc[
            self.data.ssnamenr == self.target, f"{self.model_name}_G1_{self.filter}"
        ].values[0]
        G2 = self.data.loc[
            self.data.ssnamenr == self.target, f"{self.model_name}_G2_{self.filter}"
        ].values[0]
        return H, G1, G2

    def mag_model(self, x):
        H, G1, G2 = self.get_data_model()
        return finkus.func_hg1g2(x, H, G1, G2)


class sHG1G2(Model):
    def __init__(self, band: str, target: str = None) -> None:
        super().__init__(band, target)
        self.model_name = "SHG1G2"

    def get_data_model(self):
        H = self.data.loc[
            self.data.ssnamenr == self.target, f"{self.model_name}_H_{self.filter}"
        ].values[0]
        G1 = self.data.loc[
            self.data.ssnamenr == self.target, f"{self.model_name}_G1_{self.filter}"
        ].values[0]
        G2 = self.data.loc[
            self.data.ssnamenr == self.target, f"{self.model_name}_G2_{self.filter}"
        ].values[0]
        sRA = self.data.loc[
            self.data.ssnamenr == self.target, f"{self.model_name}_alpha0"
        ].values[0]
        sDEC = self.data.loc[
            self.data.ssnamenr == self.target, f"{self.model_name}_delta0"
        ].values[0]
        R = self.data.loc[
            self.data.ssnamenr == self.target, f"{self.model_name}_R"
        ].values[0]
        return H, G1, G2, sRA, sDEC, R

    def pha_eph(self):
        spin_coord = SkyCoord(self.eph.RA, self.eph.DEC, unit=(u.hourangle, u.deg))
        pha = np.transpose(
            [
                [i, j, k]
                for i, j, k in zip(
                    np.radians(self.eph.Phase),
                    np.radians(spin_coord.ra.deg),
                    np.radians(spin_coord.dec.deg),
                )
            ]
        )
        return pha

    def pha_ztf(self):
        cond = self.ztf["i:fid"] == self.int_filter
        spin_coord = SkyCoord(
            self.ztf.loc[cond, "RA"],
            self.ztf.loc[cond, "Dec"],
            unit=(u.hourangle, u.deg),
        )
        pha = np.transpose(
            [
                [i, j, k]
                for i, j, k in zip(
                    np.radians(self.ztf.loc[cond, "Phase"]),
                    np.radians(spin_coord.ra.deg),
                    np.radians(spin_coord.dec.deg),
                )
            ]
        )
        return pha

    def mag_model(self, x):
        H, G1, G2, sRA, sDEC, R = self.get_data_model()
        return finkus.func_hg1g2_with_spin(
            x, H, G1, G2, R, np.radians(sRA), np.radians(sDEC)
        )

    def mag_dist_ztf(self):
        cond = self.ztf["i:fid"] == self.int_filter
        return self.mag_dist(
            self.pha_ztf(), (self.ztf.loc[cond, "Dobs"], self.ztf.loc[cond, "Dhelio"])
        )

    def mag_dist_eph(self):
        return self.mag_dist(self.pha_eph(), (self.eph["Dobs"], self.eph["Dhelio"]))

    def aspect_angle_extrema(self):
        ph = np.radians(self.eph.Phase)
        H, G1, G2, sRA, sDEC, R = self.get_data_model()
        g = finkus.func_hg1g2(ph, H, G1, G2)

        coords_eph = SkyCoord(self.eph.RA, self.eph.DEC, unit=(u.hourangle, u.deg))
        cosL = finkus.spin_angle(
            coords_eph.ra.radian,
            coords_eph.dec.radian,
            np.radians(sRA),
            np.radians(sDEC),
        )
        s = 2.5 * np.log10(1 - (1 - R) * np.abs(cosL))

        return g + np.min(s), g + np.max(s)
