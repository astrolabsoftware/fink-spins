def format_HG(data_hg, filters):
    # HG to data
    names_numbers = rocks.identify(data_hg.ssnamenr)
    data_hg["number"] = [nn[1] for nn in names_numbers]
    data_hg["name"] = [nn[0] for nn in names_numbers]

    for filt in filters.keys():
        print(filt)
        data_hg["HG_H_{}".format(filters[filt])] = (
            data_hg["params"].apply(lambda x: x["H_{}".format(filt)]).values
        )
        data_hg["HG_dH_{}".format(filters[filt])] = (
            data_hg["params"].apply(lambda x: x["errH_{}".format(filt)]).values
        )
        data_hg["HG_G_{}".format(filters[filt])] = (
            data_hg["params"].apply(lambda x: x["G_{}".format(filt)]).values
        )
        data_hg["HG_dG_{}".format(filters[filt])] = (
            data_hg["params"].apply(lambda x: x["errG_{}".format(filt)]).values
        )
        data_hg["HG_rms_{}".format(filters[filt])] = (
            data_hg["params"].apply(lambda x: x["rms_{}".format(filt)]).values
        )

    data_hg["minphase"] = np.rad2deg(
        data_hg["params"].apply(lambda x: x["minphase"])
    ).values
    data_hg["maxphase"] = np.rad2deg(
        data_hg["params"].apply(lambda x: x["maxphase"])
    ).values

    data_hg["n_days"] = data_hg["ndays"]
    data_hg["n_obs"] = data_hg["nmeasurements"]

    data_hg["HG_rms"] = data_hg["params"].apply(lambda x: x["rms"]).values
    data_hg["HG_chi2red"] = data_hg["params"].apply(lambda x: x["chi2red"]).values

    cond = data_hg.name.notna()
    data_hg = data_hg[cond].reset_index(drop=True)

    return data_hg


def format_HG1G2(data_hg1g2, filters):
    # HG1G2 to data
    for filt in filters.keys():
        print(filt)
        data_hg1g2["HG1G2_H_{}".format(filters[filt])] = (
            data_hg1g2["params"].apply(lambda x: x["H_{}".format(filt)]).values
        )
        data_hg1g2["HG1G2_dH_{}".format(filters[filt])] = (
            data_hg1g2["params"].apply(lambda x: x["errH_{}".format(filt)]).values
        )
        data_hg1g2["HG1G2_G1_{}".format(filters[filt])] = (
            data_hg1g2["params"].apply(lambda x: x["G1_{}".format(filt)]).values
        )
        data_hg1g2["HG1G2_dG1_{}".format(filters[filt])] = (
            data_hg1g2["params"].apply(lambda x: x["errG1_{}".format(filt)]).values
        )
        data_hg1g2["HG1G2_G2_{}".format(filters[filt])] = (
            data_hg1g2["params"].apply(lambda x: x["G2_{}".format(filt)]).values
        )
        data_hg1g2["HG1G2_dG2_{}".format(filters[filt])] = (
            data_hg1g2["params"].apply(lambda x: x["errG2_{}".format(filt)]).values
        )
        data_hg1g2["HG1G2_rms_{}".format(filters[filt])] = (
            data_hg1g2["params"].apply(lambda x: x["rms_{}".format(filt)]).values
        )

    data_hg1g2["HG1G2_rms"] = data_hg1g2["params"].apply(lambda x: x["rms"]).values
    data_hg1g2["HG1G2_chi2red"] = (
        data_hg1g2["params"].apply(lambda x: x["chi2red"]).values
    )
    data_hg1g2["HG1G2_status"] = (
        data_hg1g2["params"].apply(lambda x: x["status"]).values
    )
    data_hg1g2["HG1G2_fit"] = data_hg1g2["params"].apply(lambda x: x["fit"]).values

    data_hg1g2 = data_hg1g2.drop(columns=["nmeasurements", "ndays", "params"])

    return data_hg1g2


def format_sHG1G2(data_hg1g2hyb, filters):
    # HG1G2hyb to data
    for filt in filters.keys():
        print(filt)
        data_hg1g2hyb["HG1G2hyb_H_{}".format(filters[filt])] = (
            data_hg1g2hyb["params"].apply(lambda x: x["H_{}".format(filt)]).values
        )
        data_hg1g2hyb["HG1G2hyb_dH_{}".format(filters[filt])] = (
            data_hg1g2hyb["params"].apply(lambda x: x["errH_{}".format(filt)]).values
        )
        data_hg1g2hyb["HG1G2hyb_G1_{}".format(filters[filt])] = (
            data_hg1g2hyb["params"].apply(lambda x: x["G1_{}".format(filt)]).values
        )
        data_hg1g2hyb["HG1G2hyb_dG1_{}".format(filters[filt])] = (
            data_hg1g2hyb["params"].apply(lambda x: x["errG1_{}".format(filt)]).values
        )
        data_hg1g2hyb["HG1G2hyb_G2_{}".format(filters[filt])] = (
            data_hg1g2hyb["params"].apply(lambda x: x["G2_{}".format(filt)]).values
        )
        data_hg1g2hyb["HG1G2hyb_dG2_{}".format(filters[filt])] = (
            data_hg1g2hyb["params"].apply(lambda x: x["errG2_{}".format(filt)]).values
        )
        data_hg1g2hyb["HG1G2hyb_rms_{}".format(filters[filt])] = (
            data_hg1g2hyb["params"].apply(lambda x: x["rms_{}".format(filt)]).values
        )

    data_hg1g2hyb["HG1G2hyb_RA0"] = np.degrees(
        data_hg1g2hyb["params"].apply(lambda x: x["alpha0"]).values
    )
    data_hg1g2hyb["HG1G2hyb_dRA0"] = np.degrees(
        data_hg1g2hyb["params"].apply(lambda x: x["erralpha0"]).values
    )
    data_hg1g2hyb["HG1G2hyb_DEC0"] = np.degrees(
        data_hg1g2hyb["params"].apply(lambda x: x["delta0"]).values
    )
    data_hg1g2hyb["HG1G2hyb_dDEC0"] = np.degrees(
        data_hg1g2hyb["params"].apply(lambda x: x["errdelta0"]).values
    )
    data_hg1g2hyb["HG1G2hyb_R"] = data_hg1g2hyb["params"].apply(lambda x: x["R"]).values
    data_hg1g2hyb["HG1G2hyb_dR"] = (
        data_hg1g2hyb["params"].apply(lambda x: x["errR"]).values
    )

    data_hg1g2hyb["HG1G2hyb_rms"] = (
        data_hg1g2hyb["params"].apply(lambda x: x["rms"]).values
    )
    data_hg1g2hyb["HG1G2hyb_chi2red"] = (
        data_hg1g2hyb["params"].apply(lambda x: x["chi2red"]).values
    )
    data_hg1g2hyb["HG1G2hyb_status"] = (
        data_hg1g2hyb["params"].apply(lambda x: x["status"]).values
    )
    data_hg1g2hyb["HG1G2hyb_fit"] = (
        data_hg1g2hyb["params"].apply(lambda x: x["fit"]).values
    )
    data_hg1g2hyb["HG1G2hyb_minCosLambda"] = (
        data_hg1g2hyb["params"].apply(lambda x: x["minCosLambda"]).values
    )
    data_hg1g2hyb["HG1G2hyb_meanCosLambda"] = (
        data_hg1g2hyb["params"].apply(lambda x: x["meanCosLambda"]).values
    )
    data_hg1g2hyb["HG1G2hyb_maxCosLambda"] = (
        data_hg1g2hyb["params"].apply(lambda x: x["maxCosLambda"]).values
    )

    data_hg1g2hyb = data_hg1g2hyb.drop(columns=["nmeasurements", "ndays", "params"])

    return data_hg1g2hyb


if __name__ == "__main__":
    import numpy as np
    import pandas as pd

    import rocks

    rocks.set_log_level("error")

    # Local Configuration
    data_fink = "data/ztf/"

    # ZTF filters 1: g, 2: r
    filters = {"1": "g", "2": "r"}

    data_hg1g2hyb = pd.read_parquet(f"{data_fink}sso_fink_SHG1G2.parquet")  # All
    data_hg1g2 = pd.read_parquet(f"{data_fink}sso_fink_HG1G2.parquet")
    data_hg = pd.read_parquet(f"{data_fink}sso_fink_HG.parquet")

    print(
        "check data:\n\tlen(HG): {}\n\tlen(HG1G2): {}\n\tlen(sHG1G2): {}".format(
            len(data_hg), len(data_hg1g2), len(data_hg1g2hyb)
        )
    )

    data_hg = format_HG(data_hg, filters)
    data_hg1g2 = format_HG1G2(data_hg1g2, filters)
    data_hg1g2hyb = format_sHG1G2(data_hg1g2hyb, filters)

    # HG with HG1G2
    data_2 = data_hg.merge(data_hg1g2, on="ssnamenr")

    # (HG with HG1G2) with sHG1G2
    data = data_2.merge(data_hg1g2hyb, on="ssnamenr")

    print(
        "check data:\n\tlen(HG): {}\n\tlen(HG1G2): {}\n\tlen(sHG1G2): {}\n\tlen(final data format): {}".format(
            len(data_hg), len(data_hg1g2), len(data_hg1g2hyb), len(data)
        )
    )

    data = data.astype({"number": "Int64"})

    data.to_parquet(f"{data_fink}sso_ZTF.parquet")
