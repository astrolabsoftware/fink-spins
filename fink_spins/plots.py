import numpy as np
import matplotlib.pyplot as plt

# from astropy.coordinates import SkyCoord
# import astropy.units as u

# import fink_utils.sso.spins as finkus

from fink_spins.model import Model, sHG1G2
import os


def save_plot(model: Model, save_plot, custom_title):
    if save_plot:
        if not os.path.isdir("plots"):
            os.mkdir("plots")
        plt.savefig("plots/{}_{}.png".format(model.target, custom_title))
        plt.savefig("plots/{}_{}.pgf".format(model.target, custom_title))


def plot_lightcurve(model: Model, sp=False):
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
    plot_residuals(ax[1], model, "Date")

    plt.tight_layout()
    save_plot(model, sp, "lc_{}_band".format(model.filter))
    plt.show()


def plot_phase(model: Model, sp=False):
    # Mag vs Time
    _, ax = plt.subplots(
        2,
        1,
        figsize=(12, 12),
        sharex=True,
    )
    plot_ph(ax[0], model)
    plot_residuals(ax[1], model, "Phase")

    plt.legend()
    plt.tight_layout()
    save_plot(model, sp, "phase_{}_band".format(model.filter))
    plt.show()


def plot_lc(ax, model: Model):
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
        label="{} prediction".format(model.model_name),
    )

    ax.set_ylabel("H: Absolute Magnitude")
    ax.set_title("lightcurve")
    ax.legend()


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

    ax.set_title("phase")
    ax.invert_yaxis()

    # plot model predictions
    if isinstance(model, sHG1G2):
        x = model.pha_eph()
        y = model.mag_model(x)
        min_g, max_g = model.aspect_angle_extrema()
        ax.plot(model.eph.Phase, min_g, color="grey", zorder=-5)
        ax.plot(model.eph.Phase, max_g, color="grey", zorder=-5)
    else:
        x = np.radians(model.eph["Phase"])
        y = model.mag_model(x)

    ax.plot(
        model.eph["Phase"],
        y,
        color=Model.colors[model.int_filter - 1],
        linestyle="dotted",
    )

    ax.set_ylabel("H: Absolute Magnitude")
    ax.legend()


def plot_residuals(ax, model: Model, x_col):
    cond = model.ztf["i:fid"] == model.int_filter

    # plot residuals
    residuals = model.residuals()
    chi2red = model.get_chi2red()
    ax.scatter(
        model.ztf.loc[cond, x_col],
        residuals,
        color=Model.colors[model.int_filter - 1],
        alpha=0.1,
        marker="s",
        label=f"{model.filter} band",
    )
    ax.axhline(0, color="lightgray", zorder=-100)
    ax.set_title("residuals, chi2red={:.4f}".format(chi2red))
    ax.set_xlabel(x_col)
    ax.set_ylabel("Residuals")
    ax.legend()


def compare_model_lc(hg, hg1g2, shg1g2, band, sp=False):
    _, ax = plt.subplots(
        2,
        3,
        figsize=(15, 12),
    )

    for i, model in enumerate([hg, hg1g2, shg1g2]):
        model.reset_filter(band)
        plot_lc(ax[0][i], model)
        plot_residuals(ax[1][i], model, "Date")

    plt.legend()
    plt.tight_layout()
    save_plot(model, sp, "lc_multimodel_{}_band".format(model.filter))
    plt.show()


def compare_model_phase(hg, hg1g2, shg1g2, band, sp=False):
    _, ax = plt.subplots(
        2,
        3,
        figsize=(15, 12),
    )

    for i, model in enumerate([hg, hg1g2, shg1g2]):
        model.reset_filter(band)
        plot_ph(ax[0][i], model)
        plot_residuals(ax[1][i], model, "Phase")

    plt.legend()
    plt.tight_layout()
    save_plot(model, sp, "phase_multimodel_{}_band".format(model.filter))
    plt.show()


def plot_lc_multiband(model: Model, sp=False):
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
    for filter in Model.dict_band.keys():
        model.reset_filter(filter)
        plot_lc(ax[0], model)
        plot_residuals(ax[1], model, "Date")

    plt.tight_layout()
    save_plot(model, sp, "lc_multiband")
    plt.show()


def plot_phase_multiband(model: Model, sp=False):
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
    for filter in Model.dict_band.keys():
        model.reset_filter(filter)
        plot_ph(ax[0], model)
        plot_residuals(ax[1], model, "Phase")

    plt.tight_layout()
    save_plot(model, sp, "phase_multiband")
    plt.show()
