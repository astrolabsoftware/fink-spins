import matplotlib
import numpy as np

GOLDEN_RATIO = (np.sqrt(5) + 1) / 2 

def figsize(width=0.5, aspect=GOLDEN_RATIO):
    """Compute the figure width and height in inches.

    Parameters
    ==========
    width : float
        The figure width in units of the \textwidth. Default is 0.5,
        corresponding to one columnwidth.
    aspect : float
        The aspect ratio (width / height) of the figure. Default is the golden ratio.

    Returns
    =======
    tuple
        The figure width and height in inches.
    """
    TEXTWIDTH = 523.5307  # in points, from the LaTeX document log file
    POINTS_TO_INCHES = 1 / 72.27

    figure_width = TEXTWIDTH * POINTS_TO_INCHES * width
    figure_height = figure_width / aspect

    return figure_width, figure_height

SETUP = {
    # ----------
    # GENERAL SET-UP
    "backend"              : "pgf",         # To export figures to .pgf
    "backend_fallback"     : True,          # If not compatible,
                                            #  mpl will find a compatible one
    "toolbar"              : "toolbar2",    # "toolbar2", "toolmanager", "None"
    "timezone"             : "UTC",         # a pytz timezone string,
                                            # eg US/Central or Europe/Paris

    # ----------
    # LaTeX SET-UP
    "text.usetex"          : True,          # use inline math for ticks
    "pgf.rcfonts"          : False,         # setup fonts from rc parameters

    # Packages required for figure compilation:
    "pgf.preamble":        
                    r"\usepackage{amsmath} "
                    r"\usepackage[utf8x]{inputenc} "
                    r"\usepackage[T1]{fontenc} "
                    r"\usepackage{txfonts} " # txfonts are used by A&A Journal
                    r"\usepackage[default]{sourcesanspro} "
                    r"\usepackage{pgfplots} "
                    r"\usepgfplotslibrary{external} "
                    r"\tikzexternalize "
                    r"\usepackage{xcolor} "
                    ,

    # ----------
    # GENERAL
    # ----------
    "figure.figsize"       : figsize(width=0.5),
    "savefig.dpi"          : 400,
    "font.size"            : 10, 
    "font.family"          : "serif",
    "text.color"           : "#000000",
    "axes.facecolor"       : "#ffffff",      # axes background color
    "axes.edgecolor"       : "#000000",       # axes edge color
    "axes.linewidth"       : 0.5,           # edge linewidth
    "axes.grid"            : False,         # display grid or not
    "axes.titlesize"       : "large",  # fontsize of the axes title
    "axes.labelsize"       : "small", # fontsize of the x any y labels
    "axes.labelcolor"      : "black",
    "axes.axisbelow"       : True,   # whether axis gridlines and ticks are
                                     # below the axes elements (lines, text)
    "axes.xmargin"         : 0,
    "axes.ymargin"         : 0,
    "axes.spines.top"      : True,
    "axes.spines.right"    : True,
    "xtick.major.size"     : 4,      # major tick size in points
    "xtick.minor.size"     : 2,      # minor tick size in points
    "xtick.major.pad"      : 2,      # distance to major tick label in points
    "xtick.major.width"    : 0.5,
    "xtick.minor.width"    : 0.3,
    "xtick.minor.visible"  : True,
    "xtick.minor.pad"      : 2,    # distance to the minor tick label in points
    "xtick.color"          : "black", # color of the tick labels
    "xtick.top"            : True,
    "xtick.labelsize"      : "8",  # fontsize of the tick labels
    "xtick.direction"      : "in",     # direction: in or out
    "ytick.major.size"     : 4,      # major tick size in points
    "ytick.major.width"    : 0.5,
    "ytick.minor.width"    : 0.2,
    "ytick.minor.size"     : 2,      # minor tick size in points
    "ytick.major.pad"      : 2,      # distance to major tick label in points
    "ytick.minor.pad"      : 2,   # distance to the minor tick label in points
    "ytick.major.width"    : 0.5,
    "ytick.minor.width"    : 0.3,
    "ytick.right"          : True,
    "ytick.color"          : "black", # color of the tick labels
    "ytick.labelsize"      : "8",  # fontsize of the tick labels
    "ytick.direction"      : "in",     # direction: in or out
    "ytick.minor.visible"  : True,
    "grid.color"           : "black", # grid color
    "grid.linestyle"       : ":",      # dotted
    "grid.linewidth"       : "0.2",    # in points
    "legend.fontsize"      : "small",
    "legend.fancybox"      : False,  # if True, use a rounded box for the
                                     # legend, else a rectangle
    "lines.linewidth"      : 1.0,           # line width in points
    "lines.antialiased"    : True,          # render lines in antialised

}

matplotlib.rcParams.update(SETUP)
