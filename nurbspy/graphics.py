import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# Attempt to import imageio for video generatioj
try:
    import imageio

    IMEGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False


from cycler import cycler

COLORS_PYTHON = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]

COLORS_MATLAB = [
    "#0072BD",
    "#D95319",
    "#EDB120",
    "#7E2F8E",
    "#77AC30",
    "#4DBEEE",
    "#A2142F",
]


def set_plot_options(
    fontsize=14,
    grid=True,
    major_ticks=True,
    minor_ticks=True,
    margin=0.05,
    color_order="matlab",
    linewidth=1.25,
):
    """
    Set options for creating publication-quality figures using Matplotlib.

    This function updates the internal Matplotlib settings to better align with standards for publication-quality figures.
    Features include improved font selections, tick marks, grid appearance, and color selections.

    Parameters
    ----------
    fontsize : int, optional
        Font size for text elements in the plot. Default is 13.
    grid : bool, optional
        Whether to show grid lines on the plot. Default is True.
    major_ticks : bool, optional
        Whether to show major ticks. Default is True.
    minor_ticks : bool, optional
        Whether to show minor ticks. Default is True.
    margin : float, optional
        Margin size for axes. Default is 0.05.
    color_order : str, optional
        Color order to be used for plot lines. Options include "python" and "matlab". Default is "matlab".

    """

    if isinstance(color_order, str):
        if color_order.lower() == "default":
            color_order = COLORS_PYTHON

        elif color_order.lower() == "matlab":
            color_order = COLORS_MATLAB

    # Define dictionary of custom settings
    rcParams = {
        "text.usetex": False,
        "font.size": fontsize,
        "font.style": "normal",
        "font.family": "serif",  # 'serif', 'sans-serif', 'cursive', 'fantasy', 'monospace'
        "font.serif": ["Times New Roman"],  # ['times new roman', 'cmr10']
        "mathtext.fontset": "stix",  # ["stix", 'cm']
        "axes.edgecolor": "black",
        "axes.linewidth": 1.25,
        "axes.titlesize": fontsize,
        "axes.titleweight": "normal",
        "axes.titlepad": fontsize * 1.4,
        "axes.labelsize": fontsize,
        "axes.labelweight": "normal",
        "axes.labelpad": fontsize,
        "axes.xmargin": margin,
        "axes.ymargin": margin,
        "axes.zmargin": margin,
        "axes.grid": grid,
        "axes.grid.axis": "both",
        "axes.grid.which": "major",
        "axes.prop_cycle": cycler(color=color_order),
        "grid.alpha": 1,
        "grid.color": "#808080",  # Grey equivalent to 50% transparency
        "grid.linestyle": "-",
        "grid.linewidth": 0.5,
        "legend.borderaxespad": 1,
        "legend.borderpad": 0.6,
        "legend.edgecolor": "black",
        "legend.facecolor": "white",
        "legend.labelcolor": "black",
        "legend.labelspacing": 0.3,
        "legend.fancybox": True,
        "legend.fontsize": fontsize - 2,
        "legend.framealpha": 1.00,
        "legend.handleheight": 0.7,
        "legend.handlelength": 1.25,
        "legend.handletextpad": 0.8,
        "legend.markerscale": 1.0,
        "legend.numpoints": 1,
        "lines.linewidth": linewidth,
        "lines.markersize": 4.5,
        "lines.markeredgewidth": linewidth,
        "lines.markerfacecolor": "white",
        "xtick.direction": "in",
        "xtick.labelsize": fontsize - 1,
        "xtick.bottom": major_ticks,
        "xtick.top": major_ticks,
        "xtick.major.size": 6,
        "xtick.major.width": 1.25,
        "xtick.minor.size": 3,
        "xtick.minor.width": 0.75,
        "xtick.minor.visible": minor_ticks,
        "ytick.direction": "in",
        "ytick.labelsize": fontsize - 1,
        "ytick.left": major_ticks,
        "ytick.right": major_ticks,
        "ytick.major.size": 6,
        "ytick.major.width": 1.25,
        "ytick.minor.size": 3,
        "ytick.minor.width": 0.75,
        "ytick.minor.visible": minor_ticks,
        "savefig.dpi": 500,
    }

    # Update the internal Matplotlib settings dictionary
    mpl.rcParams.update(rcParams)


def print_installed_fonts():
    """
    Print the list of fonts installed on the system.

    This function identifies and prints all available fonts for use in Matplotlib.
    """
    fonts = mpl.font_manager.findSystemFonts(fontpaths=None, fontext="ttf")
    for font in sorted(fonts):
        print(font)


def print_rc_parameters(filename=None):
    """
    Print the current rcParams used by Matplotlib or write to file if provided.

    This function provides a quick overview of the active configuration parameters within Matplotlib.
    """
    params = mpl.rcParams
    for key, value in params.items():
        print(f"{key}: {value}")

    if filename:
        with open(filename, "w") as file:
            for key, value in params.items():
                file.write(f"{key}: {value}\n")


def savefig_in_formats(fig, path_without_extension, formats=[".png", ".svg", ".eps"], dpi=500):
    """
    Save a given Matplotlib figure in multiple file formats.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure object to be saved.
    path_without_extension : str
        The full path to save the figure excluding the file extension.
    formats : list of str, optional
        A list of string file extensions to specify which formats the figure should be saved in.
        Supported formats are ['.png', '.svg', '.pdf', '.eps'].
        Default is ['.png', '.svg'].
    dpi : int, optional
        The resolution in dots per inch with which to save the image. This parameter affects only PNG files.
        Default is 500.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> ax.plot([0, 1], [0, 1])
    >>> save_fig_in_formats(fig, "/path/to/figure/filename")

    This will save the figure as "filename.png", "filename.svg", and "filename.pdf" in the "/path/to/figure/" directory.
    """

    # Validate export formats
    allowed_formats = {".png", ".svg", ".pdf", ".eps"}
    invalid_formats = set(formats) - allowed_formats
    if invalid_formats:
        raise ValueError(
            f"Unsupported file formats: {', '.join(invalid_formats)}. Allowed formats: {', '.join(allowed_formats)}"
        )

    for ext in formats:
        if ext == ".png":
            # Save PNG files with specified DPI
            fig.savefig(f"{path_without_extension}{ext}", bbox_inches="tight", dpi=dpi)
        else:
            # Save in vector formats without DPI setting
            fig.savefig(f"{path_without_extension}{ext}", bbox_inches="tight")


def create_gif(image_folder, output_file, duration=0.5):
    """
    Create a GIF from a series of images.

    Parameters
    ----------
    image_folder : str
        The path to the folder containing the images.
    output_file : str
        The path and filename of the output GIF.
    duration : float, optional
        Duration of each frame in the GIF, by default 0.5 seconds.
    """

    if not IMEGEIO_AVAILABLE:
        raise ImportError(
            "imageio is not installed. Please run `pip install imageio-ffmpeg` to run this function."
        )

    images = []
    for filename in sorted(os.listdir(image_folder)):
        if filename.endswith(".png"):
            file_path = os.path.join(image_folder, filename)
            images.append(imageio.imread(file_path))

    imageio.mimsave(output_file, images, duration=duration)


def create_mp4(image_folder, output_file, fps=10):
    """
    Create an MP4 video from a series of images.

    Parameters
    ----------
    image_folder : str
        The path to the folder containing the images.
    output_file : str
        The path and filename of the output MP4 video.
    fps : int, optional
        Frames per second in the output video, by default 10.
    """

    if not IMEGEIO_AVAILABLE:
        raise ImportError(
            "imageio is not installed. Please run `pip install imageio-ffmpeg` to run this function."
        )

    with imageio.get_writer(output_file, fps=fps) as writer:
        for filename in sorted(os.listdir(image_folder)):
            if filename.endswith(".png"):
                file_path = os.path.join(image_folder, filename)
                image = imageio.imread(file_path)
                writer.append_data(image)


def scale_graphics_x(fig, scale, mode="multiply"):
    """Scale x-coordinates of graphics objects"""

    for ax in fig.get_axes():
        # Scaling lines
        for line in ax.get_lines():
            xdata, ydata = line.get_data()
            if mode == "multiply":
                line.set_xdata(xdata * scale)
            elif mode == "add":
                line.set_xdata(xdata + scale)

        # Scaling patches (like rectangles)
        for patch in ax.patches:
            if mode == "multiply":
                xy = patch.get_xy()
                xy[:, 0] = xy[:, 0] * scale
                patch.set_xy(xy)
            elif mode == "add":
                xy = patch.get_xy()
                xy[:, 0] = xy[:, 0] + scale
                patch.set_xy(xy)

        # Scaling contour plots
        for collection in ax.collections:
            for path in collection.get_paths():
                if mode == "multiply":
                    path.vertices[:, 0] *= scale
                elif mode == "add":
                    path.vertices[:, 0] += scale


def scale_graphics_y(fig, scale, mode="multiply"):
    """Scale y-coordinates of graphics objects"""

    for ax in fig.get_axes():
        # Scaling lines
        for line in ax.get_lines():
            xdata, ydata = line.get_data()
            if mode == "multiply":
                line.set_ydata(ydata * scale)
            elif mode == "add":
                line.set_ydata(ydata + scale)

        # Scaling patches (like rectangles)
        for patch in ax.patches:
            if mode == "multiply":
                xy = patch.get_xy()
                xy[:, 1] = xy[:, 1] * scale
                patch.set_xy(xy)
            elif mode == "add":
                xy = patch.get_xy()
                xy[:, 1] = xy[:, 1] + scale
                patch.set_xy(xy)

        # Scaling contour plots
        for collection in ax.collections:
            for path in collection.get_paths():
                if mode == "multiply":
                    path.vertices[:, 1] *= scale
                elif mode == "add":
                    path.vertices[:, 1] += scale
