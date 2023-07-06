import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u

import fink_utils.sso.spins as finkus

from fink_spins.model import Model, sHG1G2


def plot_lightcurve(model: Model):
    # Mag vs Time
    _, ax = plt.subplots(
        2,
        1,
        figsize=(12, 12),  # fs.figsize(1),
        sharex=True,
        # gridspec_kw={
        #     'top':0.995,
        #     'left':0.075,
        #     'right':0.995,
        #     'bottom':0.085,
        #     'hspace':0.02,
        #     'height_ratios': [2,1]}
    )
    plot_lc(ax[0], model)
    ax[0].set_title("lightcurve")

    plot_residuals(ax[1], model, "Date")
    ax[1].set_title("residuals")

    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_phase(model: Model):
    # Mag vs Time
    _, ax = plt.subplots(
        2,
        1,
        figsize=(12, 12),  # fs.figsize(1),
        sharex=True,
        # gridspec_kw={
        #     'top':0.995,
        #     'left':0.075,
        #     'right':0.995,
        #     'bottom':0.085,
        #     'hspace':0.02,
        #     'height_ratios': [2,1]}
    )
    plot_ph(ax[0], model)

    ax[0].set_title("lightcurve")
    ax[0].invert_yaxis()

    plot_residuals(ax[1], model, "Phase")
    ax[1].set_title("residuals")

    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_lc(ax, model):
    cond = model.ztf["i:fid"] == model.int_filter

    # plot observations
    ax.scatter(
        model.ztf.loc[cond, "Date"],
        model.ztf.loc[cond, "i:magpsf"],
        s=20,
        marker="o",
        color=Model.colors[model.int_filter - 1],
        label=f"ZTF {model.filter}",
    )

    # plot model predictions
    pred_mag = model.mag_dist_eph()
    ax.plot(
        model.eph["Date"],
        pred_mag,
        color=Model.colors[model.int_filter - 1],
        linestyle="dotted",
        label="HG",
    )


def plot_ph(ax, model):
    cond = model.ztf["i:fid"] == model.int_filter

    # plot observations
    # fmt: off
    ax.scatter(
        model.ztf.loc[cond, "Phase"],
        model.ztf.loc[cond, "i:magpsf"] - Model.dist_reduction(
            model.ztf.loc[cond, "Dobs"], model.ztf.loc[cond, "Dhelio"]
        ),
        s=20,
        marker="o",
        color=Model.colors[model.int_filter - 1],
        label=f"ZTF {model.filter}",
    )
    # fmt: on

    # plot model predictions
    if isinstance(model, sHG1G2):
        x = model.pha_eph()
        y = model.mag_model(x)
        _, G1, G2, sRA, sDEC, _ = model.get_data_model()

        # Compute Lambda
        coords = SkyCoord(
            model.ztf["i:ra"], model.ztf["i:dec"], unit=(u.hourangle, u.deg)
        )

        ra = np.radians(coords.ra.deg)
        dec = np.radians(coords.dec.deg)
        cos_lambda = np.sin(dec) * np.sin(sDEC) + np.cos(dec) * np.cos(sDEC) * np.cos(
            ra - sRA
        )
        idx_mincos = np.argmin(cos_lambda)
        idx_maxcos = np.argmax(cos_lambda)
        x_min = [x[0, idx_mincos], x[1, idx_mincos], x[2, idx_mincos]]
        x_max = [x[0, idx_maxcos], x[1, idx_maxcos], x[2, idx_maxcos]]
        min_hg1g2 = finkus.func_hg1g2(
            np.radians(model.eph["Phase"]), model.mag_model(x_min), G1, G2
        )
        max_hg1g2 = finkus.func_hg1g2(
            np.radians(model.eph["Phase"]), model.mag_model(x_min), G1, G2
        )
        # ax[0].plot(
        #     model.eph["Phase"],
        #     min_hg1g2,
        #     color=Model.colors[model.int_filter - 1],
        #     linestyle="dotted",
        #     label="min H",
        # )
        # ax[0].plot(
        #     model.eph["Phase"],
        #     max_hg1g2,
        #     color=Model.colors[model.int_filter - 1],
        #     linestyle="dotted",
        #     label="max H",
        # )
    else:
        x = np.radians(model.eph["Phase"])
        y = model.mag_model(x)

    ax.plot(
        model.eph["Phase"],
        y,
        color=Model.colors[model.int_filter - 1],
        linestyle="dotted",
        label="HG",
    )


def plot_residuals(ax, model, x_col):
    cond = model.ztf["i:fid"] == model.int_filter

    # plot residuals
    residuals = model.residuals()
    ax.scatter(
        model.ztf.loc[cond, x_col],
        residuals,
        color=Model.colors[model.int_filter - 1],
        alpha=0.1,
        marker="s",
        label="HG",
    )


def compare_model_lc(hg, hg1g2, shg1g2, band):
    _, ax = plt.subplots(
        2,
        3,
        figsize=(15, 12),  # fs.figsize(1),
        # gridspec_kw={
        #     'top':0.995,
        #     'left':0.075,
        #     'right':0.995,
        #     'bottom':0.085,
        #     'hspace':0.02,
        #     'height_ratios': [2,1]}
    )

    for i, model in enumerate([hg, hg1g2, shg1g2]):
        model.reset_filter(band)
        plot_lc(ax[0][i], model)
        plot_residuals(ax[1][i], model, "Date")

    plt.legend()
    plt.tight_layout()
    plt.show()


def compare_model_phase(hg, hg1g2, shg1g2, band):
    _, ax = plt.subplots(
        2,
        3,
        figsize=(15, 12),  # fs.figsize(1),
        # gridspec_kw={
        #     'top':0.995,
        #     'left':0.075,
        #     'right':0.995,
        #     'bottom':0.085,
        #     'hspace':0.02,
        #     'height_ratios': [2,1]}
    )

    for i, model in enumerate([hg, hg1g2, shg1g2]):
        model.reset_filter(band)
        plot_ph(ax[0][i], model)
        plot_residuals(ax[1][i], model, "Phase")

    plt.legend()
    plt.tight_layout()
    plt.show()
