import numpy as np
import matplotlib.pyplot as plt

import fink_utils.sso.spins as finkus

from model import Model, sHG1G2


def plot_lightcurve(model: Model):
    # Mag vs Time
    fig, ax = plt.subplots(
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
    cond = model.ztf["i:fid"] == model.int_filter

    # plot observations
    ax[0].scatter(
        model.ztf.loc[cond, "Date"],
        model.ztf.loc[cond, "i:magpsf"],
        s=20,
        marker="o",
        color=Model.colors[model.int_filter - 1],
        label=f"ZTF {model.filter}",
    )

    # plot model predictions
    pred_mag = model.mag_dist_eph()
    ax[0].plot(
        model.eph["Date"],
        pred_mag,
        color=Model.colors[model.int_filter - 1],
        linestyle="dotted",
        label="HG",
    )
    ax[0].set_title("lightcurve")

    # plot residuals
    residuals = model.residuals()
    ax[1].scatter(
        model.ztf.loc[cond, "Date"],
        residuals,
        color=Model.colors[model.int_filter - 1],
        alpha=0.1,
        marker="s",
        label="HG",
    )
    ax[1].set_title("residuals")

    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_phase(model: Model):
    # Mag vs Time
    fig, ax = plt.subplots(
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
    cond = model.ztf["i:fid"] == model.int_filter

    # plot observations
    # fmt: off
    ax[0].scatter(
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
        _, G1, G2, _, _, _ = model.get_data_model()
        min_hg1g2 = finkus.func_hg1g2(np.radians(model.eph["Phase"]), np.min(y), G1, G2)
        max_hg1g2 = finkus.func_hg1g2(np.radians(model.eph["Phase"]), np.max(y), G1, G2)
        ax[0].plot(
            model.eph["Phase"],
            min_hg1g2,
            color=Model.colors[model.int_filter - 1],
            linestyle="dotted",
            label="min H",
        )
        ax[0].plot(
            model.eph["Phase"],
            max_hg1g2,
            color=Model.colors[model.int_filter - 1],
            linestyle="dotted",
            label="max H",
        )
    else:
        x = np.radians(model.eph["Phase"])
        y = model.mag_model(x)

    ax[0].plot(
        model.eph["Phase"],
        y,
        color=Model.colors[model.int_filter - 1],
        linestyle="dotted",
        label="HG",
    )
    ax[0].set_title("lightcurve")
    ax[0].invert_yaxis()

    # plot residuals
    residuals = model.residuals()
    ax[1].scatter(
        model.ztf.loc[cond, "Phase"],
        residuals,
        color=Model.colors[model.int_filter - 1],
        alpha=0.1,
        marker="s",
        label="HG",
    )
    ax[1].set_title("residuals")

    plt.legend()
    plt.tight_layout()
    plt.show()
