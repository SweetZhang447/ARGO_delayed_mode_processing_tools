"""
graphs_nc.py — Interactive visualization and QC flagging tools for ARGO float data.

This module provides matplotlib-based graphs used during delayed-mode quality control
of ARGO float profiles. Functions fall into three categories:

  1. OVERVIEW GRAPHS — display all profiles at once for dataset-level inspection:
       pres_v_var_all, TS_graph_single_dataset_all_profile, TS_graph_double,
       deep_section_var_all

  2. INTERACTIVE QC FLAGGING — allow the user to click points and assign QC flags
     (1=good, 2=prob good, 3=prob bad, 4=bad) for a single profile:
       flag_point_data_graphs, flag_range_data_graphs, flag_TS_data_graphs,
       single_prof_datasnapshot

  3. THERMAL MASS / CELL THERMAL MASS (CTM) ANALYSIS — specific to RBR-equipped floats:
       make_thermal_inertia_graph, read_csv_file_for_thermal_inertia_graph_with_timestamps,
       make_thermal_inertia_graph_with_timestamps, compute_cellTM, make_der_graph

Helper utilities:
  density_inversion_test, merge_ranges, del_bad_points

All Julian days are referenced to 1950-01-01 00:00:00 UTC.
QC color convention across all interactive graphs:
  green=good(1), aqua=prob good(2), orange=prob bad(3), red=bad(4)
  fuchsia edge = density inversion flag
"""

import itertools
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import gsw
from matplotlib.collections import LineCollection
import datetime
from matplotlib.dates import date2num, DateFormatter
import mplcursors
import copy
from tools import from_julian_day, read_intermediate_nc_file, to_julian_day
from matplotlib.lines import Line2D

def TS_graph_double(df_SALs, df_TEMPs, df_JULD, df_LONs, df_LATs, df_PRESs,
        df_LATs_2, df_LONs_2, df_JULDs_2, df_PRESs_2, df_PSALs_2, df_TEMPs_2,
        float_name_1, float_name_2):
    """
    Plot a Temperature-Salinity (TS) diagram comparing two Argo float datasets.

    Each profile is drawn as a line segment colored by date. Float 1 uses a purple
    colormap; float 2 uses red. Sigma-t density contours are plotted in the background.

    Parameters
    ----------
    df_SALs, df_TEMPs : ndarray (n_profiles, n_levels)
        Salinity and temperature arrays for float 1 (reference dataset).
    df_JULD : ndarray (n_profiles,)
        Julian days for float 1 profiles, referenced to 1950-01-01.
    df_LONs, df_LATs : ndarray (n_profiles,)
        Longitude and latitude for float 1 profiles.
    df_PRESs : ndarray (n_profiles, n_levels)
        Pressure for float 1 (currently unused; reserved for SA/CT conversions).
    df_LATs_2, df_LONs_2, df_JULDs_2, df_PRESs_2, df_PSALs_2, df_TEMPs_2 : ndarray
        Equivalent arrays for float 2 (comparison dataset).
    float_name_1, float_name_2 : str
        Float identifiers used in the plot title (e.g. 'F9443', 'F9186').
    """
    # Get rid of data conversions for QC
    # for i in np.arange(df_SALs.shape[0]):    # number of profiles
    #     df_SALs[i, :] = gsw.conversions.SA_from_SP(df_SALs[i, :], df_TEMPs[i, :], df_LONs[i], df_LATs[i])
    #     df_TEMPs[i, :] = gsw.conversions.CT_from_t(df_SALs[i, :], df_TEMPs[i, :], df_PRESs[i, :])

    # for i in np.arange(df_PSALs_2.shape[0]):    # number of profiles
    #     df_PSALs_2[i, :] = gsw.conversions.SA_from_SP(df_PSALs_2[i, :], df_TEMPs_2[i, :], df_LONs_2[i], df_LATs_2[i])
    #     df_TEMPs_2[i, :] = gsw.conversions.CT_from_t(df_PSALs_2[i, :], df_TEMPs_2[i, :], df_PRESs_2[i, :])

    # Define salinity and temperature bounds for the contour plot based on ref dataset
    smin = np.nanmin(df_SALs) - 1
    smax = np.nanmax(df_SALs) + 1
    tmin = np.nanmin(df_TEMPs) - 1
    tmax = np.nanmax(df_TEMPs) + 1

    # Calculate number of grid cells needed in the x and y dimensions
    xdim = int(np.ceil((smax - smin) / 0.1))
    ydim = int(np.ceil((tmax - tmin)))
    dens = np.zeros((ydim, xdim))

    # Create temp and salt vectors of appropriate dimensions
    ti = np.linspace(0, ydim, ydim) + tmin
    si = np.linspace(1, xdim, xdim) * 0.1 + smin

    # Loop to fill in grid with densities
    for j in range(ydim):
        for i in range(xdim):
            dens[j, i] = gsw.rho(si[i], ti[j], 0)

    # Subtract 1000 to convert to sigma-t
    dens = dens - 1000
    
    # Prepare F9443 dataset points
    ref_segments = np.stack((df_SALs, df_TEMPs), axis=2)
    # Create a LineCollection for F9443
    ref_lc = LineCollection(ref_segments, array = df_JULD, cmap='Purples', alpha=0.7)
    ref_lc.set_linewidth(1)

    # Prepare F9186 dataset points
    F9186_segments = np.stack((df_PSALs_2, df_TEMPs_2), axis=2)
    # Create a LineCollection for F9186
    F9186_lc = LineCollection(F9186_segments, array = df_JULDs_2, cmap='Reds', alpha=0.5)
    F9186_lc.set_linewidth(1)

    # Plot data
    fig, ax = plt.subplots(figsize=(10,6))
    CS = plt.contour(si, ti, dens, linestyles='dashed', colors='k')
    plt.clabel(CS, fontsize=12, inline=1, fmt='%.2f')

    # Add LineCollections to the plot
    ax = plt.gca()
    ax.add_collection(ref_lc)
    ax.add_collection(F9186_lc)

    # Add colorbar with date formatter
    cbar = plt.colorbar(ref_lc, ax=ax, label='Date')
    date_formatter = DateFormatter('%Y-%m-%d')

    # Convert colorbar ticks to regular dates
    cbar_ticks = [datetime.datetime(1950, 1, 1) + (datetime.timedelta(days=float(juld))) for juld in cbar.get_ticks()]
    cbar.ax.yaxis.set_major_formatter(date_formatter)
    cbar.set_ticklabels([dt.strftime('%Y-%m-%d') for dt in cbar_ticks])

    plt.xlim([smin + 0.75, smax - 0.75])
    plt.ylim([tmin + 0.75, tmax - 0.75])
    plt.xlabel('Salinity (PSU)')
    plt.ylabel('In-Situ Temperature (degC)')
    plt.title(f"Argo Float {float_name_1} (RED) v {float_name_2} TS Graph")

    # Adjust spacing, show plot
    plt.tight_layout()
    plt.show()

def pres_v_var_all(df_PRESs, df_VARs, df_JULD, df_prof_nums, compare_var, float_name):
    """
    Interactive pressure-vs-variable overview graph for all profiles.

    Each profile is plotted as a colored line (jet colormap, by date). Hovering
    over a line shows its profile number and date. Clicking a line toggles its
    inclusion in the returned selection set.

    Parameters
    ----------
    df_PRESs : ndarray (n_profiles, n_levels)
        Pressure values for all profiles.
    df_VARs : ndarray (n_profiles, n_levels)
        Variable values (TEMP or PSAL) for all profiles.
    df_JULD : ndarray (n_profiles,)
        Julian days for each profile, referenced to 1950-01-01.
    df_prof_nums : ndarray (n_profiles,)
        Profile numbers for each profile.
    compare_var : str
        Either 'PSAL' or 'TEMP' — sets the x-axis label.
    float_name : str
        Float identifier used in the plot title.

    Returns
    -------
    selected_profiles : set of int
        Profile numbers clicked/selected by the user before closing the window.
    """
    # Make graph
    fig, ax = plt.subplots()
   
    # Normalize df_JULD for color mapping
    norm = plt.Normalize(vmin=df_JULD.min(), vmax=df_JULD.max())
    cmap = plt.get_cmap('jet')

    # Store profile nums
    selected_profiles = set()
    # Store lines for cursor connection
    lines = []
    hovered_line = None

    # Plot each profile individually
    for i in range(df_PRESs.shape[0]):
        color = cmap(norm(df_JULD[i]))  # Assign color based on date
        line, = ax.plot(df_VARs[i, :], df_PRESs[i, :], color=color, alpha=0.7)
        line.profile_number = df_prof_nums[i]  # Attach profile number to the line
        line.juld_date = datetime.datetime(1950, 1, 1) + datetime.timedelta(days=float(df_JULD[i]))  # Attach date
        lines.append(line)

    # Invert y-axis and add grid
    plt.gca().invert_yaxis()
    plt.grid(visible=True)

    # Add colorbar with date formatter
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # ScalarMappable needs this, even if not used directly

    cbar = plt.colorbar(sm, ax=ax, label='Date')
    # Set the tick locations and labels for the colorbar based on df_JULD
    cbar_ticks = np.linspace(df_JULD.min(), df_JULD.max(), num=5)
    cbar.set_ticks(cbar_ticks)
    # Convert colorbar ticks to regular dates
    cbar_labels = [datetime.datetime(1950, 1, 1) + (datetime.timedelta(days=float(juld)))  for juld in cbar.get_ticks()]
    cbar.ax.set_yticklabels([dt.strftime('%Y-%m-%d') for dt in cbar_labels])

    # Use mplcursors to display profile numbers on hover
    cursor = mplcursors.cursor(lines, hover=True)
    # Attach profile numbers to each segment
    def annotate_hover(sel):
        nonlocal hovered_line
        hovered_line = sel.artist  # Get the line that was hovered over
        profile_number = hovered_line.profile_number  # Retrieve the attached profile number
        juld_date = hovered_line.juld_date.strftime('%Y-%m-%d')  # Format the date
        sel.annotation.set_text(f"Profile: {profile_number}\nDate: {juld_date}")
    cursor.connect("add", annotate_hover)

    # Add click event
    def on_click(event):
        nonlocal selected_profiles, hovered_line
        if hovered_line:
            profile_number = hovered_line.profile_number
            if profile_number in selected_profiles:
                selected_profiles.remove(profile_number)
                print(f"Removing {profile_number}")
            else:
                selected_profiles.add(profile_number)
                print(f"Adding {profile_number}")
            print(f"Selected Profiles: {selected_profiles}")
            print("=====================" + "====" * len(selected_profiles))
    fig.canvas.mpl_connect('button_press_event', on_click)

    # Set axis labels
    plt.ylabel("Pressure")
    if compare_var == 'PSAL':
        plt.xlabel("Salinity")
    elif compare_var == 'TEMP':
        plt.xlabel("Temperature")
    else:
        raise ValueError("Please specify 'compare_var' as 'PSAL' or 'TEMP'")

    # Set title
    plt.title(f"Argo Float {float_name} PRES-{compare_var} Graph")
    plt.show()

    return selected_profiles

def flag_point_data_graphs(var, PRES, data_type, qc_arr, profile_num, date, argodata=None, ax=None, figure=None, run_inversion=True):
    """
    Interactive point-by-point QC flagging graph (PRES v variable, or PRES v index).

    Each data point is color-coded by its current QC flag. Clicking a point cycles
    it through the four QC states: bad(red) → prob bad(orange) → prob good(aqua) → good(green).
    Points involved in a density inversion are highlighted with a fuchsia edge color.

    Can be embedded in a multi-panel figure (pass ax and figure) or shown standalone.

    Parameters
    ----------
    var : ndarray (n_levels,)
        Variable data (TEMP or PSAL) for a single profile.
    PRES : ndarray (n_levels,)
        Pressure values for a single profile.
    data_type : str
        'PRES', 'PSAL', or 'TEMP'. Determines axis labels and plot type.
        If 'PRES', x-axis shows depth index instead of variable value.
    qc_arr : ndarray (n_levels,)
        Initial QC flags for each point (1=good, 2=prob good, 3=prob bad, 4=bad).
    profile_num : int
        Profile number, used in the plot title.
    date : float
        Julian day of the profile (referenced to 1950-01-01).
    argodata : dict, optional
        Full intermediate netCDF data dict (from read_intermediate_nc_file).
        Required if data_type is 'PSAL' in order to run the density inversion test.
    ax : matplotlib.axes.Axes, optional
        Axis to plot on; if None, a new figure/axis is created.
    figure : matplotlib.figure.Figure, optional
        Parent figure when using a pre-existing axis.
    run_inversion : bool, optional
        If True (default), run and display the density inversion test.

    Returns
    -------
    selected_points : list of int
        Updated QC flags for each point after user interaction.
    """
    print_multiplot = False
    if ax is None:
        # Create the figure and axes
        fig, ax = plt.subplots()
    else:
        print_multiplot = True
        fig = figure

    if argodata is None and data_type == "PSAL":
        raise Exception("Need argo_data param if data type is PSAL to run density inversion tests")
    
    colors = []
    selected_points = []
    edge_colors = []
    exclude = []
    for qc in qc_arr:
        if qc == 4:                 # bad
            colors.append('red')
            selected_points.append(4)
            edge_colors.append('red')
        elif qc == 3:               # prob bad
            colors.append('orange')
            selected_points.append(3)
            edge_colors.append('orange')
        elif qc == 2:               # prob good
            colors.append('aqua')
            selected_points.append(2)
            edge_colors.append('aqua')
        else:                       # qc == 1, val is good
            colors.append('green')
            selected_points.append(1)
            edge_colors.append('green')
    
    # Inversion test
    if run_inversion is True:
        if argodata is not None:
            inversion = density_inversion_test(argodata, profile_num)
            for i in inversion:
                edge_colors[i] = 'fuchsia'

    # Plot good points (green)
    if data_type == "PRES":
        # Scatter plot for good and bad points
        scatter_plt = ax.scatter(np.arange(len(PRES)), PRES, color=colors, s=35, alpha=0.9, label='Good Data')
    else:
        ax.plot(var, PRES, color='blue', linewidth=2)
        scatter_plt = ax.scatter(var, PRES, color=colors, edgecolors=edge_colors, s=35, alpha=0.9)

    # Invert y-axis and add grid
    ax.invert_yaxis()
    ax.grid(visible=True)

    # Hover functionality
    cursor = mplcursors.cursor([scatter_plt], hover=True)
    def annotate_hover(sel):
        x, y = sel.target
        sel.annotation.set_text(f"{data_type}: {x:.2f}\nPRES: {y}")
    cursor.connect("add", annotate_hover)

    # Click event
    def on_click(event):
        nonlocal selected_points, argodata, profile_num, exclude
        for scatter, graph_colors, edgecolor in [(scatter_plt, colors, edge_colors)]:
            cont, ind = scatter.contains(event)
            if cont:
                clicked_idx = ind["ind"][0] 
                # Get org color of point
                org_color = graph_colors[clicked_idx]
                # Check to see if clicked point was flagged as density inversion
                run_density_inversion = False
                if argodata is not None:
                    if edgecolor[clicked_idx] == 'fuchsia':
                        run_density_inversion = True
                # Cycle through color options
                if org_color == 'red':
                    graph_colors[clicked_idx] = 'orange'
                    selected_points[clicked_idx] = 3
                    edge_colors[clicked_idx] = 'orange'
                elif org_color == 'orange':
                    graph_colors[clicked_idx] = 'aqua'
                    selected_points[clicked_idx] = 2
                    edge_colors[clicked_idx] = 'aqua'
                elif org_color == 'aqua':
                    graph_colors[clicked_idx] = 'green'
                    selected_points[clicked_idx] = 1
                    edge_colors[clicked_idx] = 'green'
                else:
                    graph_colors[clicked_idx] = 'red'
                    selected_points[clicked_idx] = 4
                    edge_colors[clicked_idx] = 'red'

                if run_density_inversion == True:
                    # see what color we've landed on
                    # if we've marked a point as bad, we need to check it again
                    if graph_colors[clicked_idx] == 'orange' or graph_colors[clicked_idx] == 'red':
                        # add it to exclude list
                        exclude.append(clicked_idx)
                        inversion = density_inversion_test(argodata, profile_num, exclude_pts=exclude)
                        for i in inversion:
                            edgecolor[i] = 'fuchsia'

                # Update the color of the clicked point
                scatter.set_color(graph_colors)
                scatter.set_edgecolors(edgecolor)
                fig.canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event', on_click)
    
    if print_multiplot == False:
        # Custom legend elements
        if argodata is not None:
            custom_legend = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='aqua', markersize=10),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='white', markeredgecolor='fuchsia', markersize=9) 
            ]
            # Add legend to the plot
            ax.legend(
                custom_legend,
                ["Bad", "Probably Bad", "Probably Good", "Good", "Density Inversion"],  # Custom labels
                loc='lower left', title="Data Quality"
            )
        else: 
            custom_legend = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='aqua', markersize=10),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10),
            ]
            # Add legend to the plot
            ax.legend(
                custom_legend,
                ["Bad", "Probably Bad", "Probably Good", "Good"],  # Custom labels
                loc='lower left', title="Data Quality"
            )

    if data_type == "PRES":
        # Add labels and title
        ax.set_xlabel("Index")
        ax.set_ylabel('Pressure')
        if print_multiplot == False:
            ax.set_title(f"Flag Point Pressure Graph for Profile: {profile_num} on {from_julian_day(float(date)).date()}")
    else:
        # Add labels and title
        ax.set_xlabel(data_type)
        ax.set_ylabel('Pressure')
        if print_multiplot == False:
            ax.set_title(f"Flag Point PRES v {data_type} for Profile: {profile_num} on {from_julian_day(float(date)).date()}")
    
    if print_multiplot == False:
        plt.show()

    return selected_points

def density_inversion_test(argo_data, prof_num, exclude_pts=None):
    """
    Detect density inversions in a single profile.

    Converts practical salinity to Absolute Salinity and in-situ temperature
    to Conservative Temperature, then computes in-situ density. An inversion
    occurs where density decreases with increasing pressure (depth).

    Points flagged as bad (QC=4) and any explicitly excluded indices are set
    to NaN before the calculation so they don't contribute to inversion detection.

    Parameters
    ----------
    argo_data : dict
        Full intermediate netCDF data dict (from read_intermediate_nc_file).
        Must contain PSAL_ADJUSTED, TEMP_ADJUSTED, PRES_ADJUSTED and their QC
        arrays, as well as LATs and LONs.
    prof_num : int
        Profile number to test.
    exclude_pts : list of int, optional
        Indices to exclude from the inversion check (e.g. points already flagged
        as bad by the user during interactive QC).

    Returns
    -------
    failed_idxs : list of int
        Sorted list of depth-level indices involved in at least one density inversion.
    """
    i = np.where(argo_data["PROFILE_NUMS"] == prof_num)[0][0]
    psal = copy.deepcopy(argo_data["PSAL_ADJUSTED"][i])
    temp = copy.deepcopy(argo_data["TEMP_ADJUSTED"][i])
    pres = copy.deepcopy(argo_data["PRES_ADJUSTED"][i])
    # apply QC to data
    psal[np.where(argo_data["PSAL_ADJUSTED_QC"][i] == 4)] = np.nan
    temp[np.where(argo_data["TEMP_ADJUSTED_QC"][i] == 4)] = np.nan
    pres[np.where(argo_data["PRES_ADJUSTED_QC"][i] == 4)] = np.nan
    lat = argo_data["LATs"][i]
    lon = argo_data["LONs"][i]

    # Get rid of excluded points
    if exclude_pts is not None:
        for i in exclude_pts:
            psal[i] = np.nan
            temp[i] = np.nan
            pres[i] = np.nan

    # Convert PSAL to Absolute Salinity
    abs_sal = gsw.SA_from_SP(psal, pres, lon, lat)
    # Convert TEMP to Conservative Temperature
    cons_temp = gsw.CT_from_t(abs_sal, temp, pres)
    # Calculate in-situ density
    dens = gsw.rho(abs_sal, cons_temp, pres)
 
    failed_idxs = []
    valid_idxs = np.where(~np.isnan(dens))[0]   # gets only valid dens
    for j in range(len(valid_idxs) - 1):
            idx1 = valid_idxs[j]
            idx2 = valid_idxs[j + 1]
            if dens[idx2] < dens[idx1]:
                # Mark both points involved in inversion
                failed_idxs.extend([idx1, idx2])

    failed_idxs = sorted(set(failed_idxs))  # Remove duplicates and sort
    print(f"Failed density inversion at indices: {failed_idxs}")

    return failed_idxs

def merge_ranges(ranges):
    """
    Merge a list of overlapping or adjacent integer index ranges.

    Parameters
    ----------
    ranges : list of (int, int)
        List of (start, end) tuples representing closed index ranges.

    Returns
    -------
    merged : list of (int, int)
        Sorted list of non-overlapping merged ranges.

    Example
    -------
    merge_ranges([(1, 3), (2, 5), (7, 9)]) -> [(1, 5), (7, 9)]
    """
    # Sort the ranges by their start values
    sorted_ranges = sorted(ranges, key=lambda x: x[0])
    
    # Initialize an empty list to store merged ranges
    merged = []
    
    for start, end in sorted_ranges:
        # If merged is empty or the current range does not overlap with the last range in merged
        if not merged or merged[-1][1] < start:
            merged.append((start, end))
        else:
            # Merge the current range with the last range in merged
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged

def flag_range_data_graphs(var, PRES, data_type, qc_arr, profile_num, date):
    """
    Interactive range-based QC flagging graph, PRES vs variable.

    The user selects a contiguous range of points by clicking two boundary points.
    All points in the range adopt the QC color of the first clicked point. Clicking
    within an existing range removes it, and clicking the same point twice cycles
    its color. Ranges are automatically merged when they overlap.

    QC color cycle on first click: bad(red) → prob bad(orange) → prob good(aqua) → good(green).

    Parameters
    ----------
    var : ndarray (n_levels,)
        Variable data (TEMP or PSAL) for a single profile.
    PRES : ndarray (n_levels,)
        Pressure values for a single profile.
    data_type : str
        'PRES', 'PSAL', or 'TEMP'. Determines axis labels and plot type.
    qc_arr : ndarray (n_levels,)
        Initial QC flags for each point (1=good, 2=prob good, 3=prob bad, 4=bad).
    profile_num : int
        Profile number, used in the plot title.
    date : float
        Julian day of the profile (referenced to 1950-01-01).

    Returns
    -------
    colors : list of str
        Final color for each point after user interaction. Corresponds to QC flags:
        'green'=1, 'aqua'=2, 'orange'=3, 'red'=4.
    """
    # Create the figure and axes
    fig, ax = plt.subplots()

    # init colors in arr
    colors = []
    selected_points = [] 
    for qc in qc_arr:
        if qc == 4:                 # bad
            colors.append('red')
        elif qc == 3:               # prob bad
            colors.append('orange')
        elif qc == 2:               # prob good
            colors.append('aqua')
        else:                       # qc == 1, val is good
            colors.append('green')
    # copy org color arr
    org_colors = copy.deepcopy(colors)
  
    # Plot points
    if data_type == 'PRES':
        scatter_plt = ax.scatter(np.arange(len(PRES)), PRES, color=colors, s=35, alpha=0.9)
    else:
        ax.plot(var, PRES, color='blue', linewidth=2)  # Line plot
        scatter_plt = ax.scatter(var, PRES, color=colors, s=35, alpha=0.9)

    # Invert y-axis and add grid
    plt.gca().invert_yaxis()
    plt.grid(visible=True)

    # Hover functionality
    cursor = mplcursors.cursor([scatter_plt], hover=True)
    def annotate_hover(sel):
        x, y = sel.target
        sel.annotation.set_text(f"{data_type}: {x:.2f}\nPRES: {y}")
    cursor.connect("add", annotate_hover)

    # Click event
    def on_click(event):
        # Tells function to use a nonlocal selected points 
        nonlocal selected_points, org_colors
        for scatter, subset_colors in [(scatter_plt, colors)]:
            cont, ind = scatter.contains(event)
            if cont:
                clicked_idx = ind["ind"][0]  
                complete_incomplete_pair = False

                for i, (p1, p2) in enumerate(selected_points):
                    # Check if clicked point is in range of a pair
                    if p1 is not None and p2 is not None:
                        if clicked_idx in np.arange(p1, p2 + 1):
                            # If it is, remove the entire range of color and remove the pair
                            selected_points.pop(i)
                            for val_in_range in np.arange(p1, p2 + 1):
                                # set color back to original color 
                                subset_colors[val_in_range] = org_colors[val_in_range]
                            # Set color and update 
                            scatter.set_color(subset_colors)
                            fig.canvas.draw_idle()
                            # exit after removal
                            print(f"Deleting range: {p1} - {p2}")
                            return 
                # If here, then clicked point is not in range of pair
                # Add the clicked point to a new or incomplete pair
                if not selected_points or selected_points[-1][1] is not None:
                    # Change color of clicked element 
                    if org_colors[clicked_idx] == 'red':
                        subset_colors[clicked_idx] = 'orange'
                    elif org_colors[clicked_idx] == 'orange':
                        subset_colors[clicked_idx] = 'aqua'
                    elif org_colors[clicked_idx] == 'aqua':
                        subset_colors[clicked_idx] = 'green'
                    else:
                        subset_colors[clicked_idx] = 'red'
                    selected_points.append((clicked_idx, None))
                    print(f"Starting new pair")
                else:
                    # Complete the last incomplete pair
                    # Check to see if it is the same
                    if clicked_idx == selected_points[-1][0]:
                        # If it is the same, cycle color options
                        if subset_colors[clicked_idx] == 'red':
                            subset_colors[clicked_idx] = 'orange'
                        elif subset_colors[clicked_idx] == 'orange':
                            subset_colors[clicked_idx] = 'aqua'
                        elif subset_colors[clicked_idx] == 'aqua':
                            subset_colors[clicked_idx] = 'green'
                        elif subset_colors[clicked_idx] == 'green':
                            subset_colors[clicked_idx] = 'red'
                        # Delete point when color has cycled to orginal color
                        if org_colors[clicked_idx] == subset_colors[clicked_idx]:
                            # remove that val
                            selected_points.pop(-1)
                            print(f"Point deleted")
                        # Update the color of the clicked point
                        scatter.set_color(subset_colors)
                        fig.canvas.draw_idle()
                        return
                    # Check to see if clicked val is before or after the last incomplete pair val
                    # If it happened after
                    elif clicked_idx > selected_points[-1][0]:
                        complete_incomplete_pair = True
                        # add the point and color as normal
                        selected_points[-1] = (selected_points[-1][0], clicked_idx)
                        # Change color
                        for val_in_range in np.arange(selected_points[-1][0], selected_points[-1][1] + 1):
                            subset_colors[val_in_range] = subset_colors[selected_points[-1][0]]
                        print(f"Completing pair: {selected_points[-1]}")
                    else:
                        complete_incomplete_pair = True
                        # switch the pair around and color as normal
                        selected_points[-1] = (clicked_idx, selected_points[-1][0])
                        # Change color
                        for val_in_range in np.arange(selected_points[-1][0], selected_points[-1][1] + 1):
                            subset_colors[val_in_range] = subset_colors[selected_points[-1][1]]
                        print(f"Completing pair: {selected_points[-1]}")

                # # check to see if we can merge any ranges 
                if complete_incomplete_pair == True:
                    selected_points_copy = copy.deepcopy(selected_points) # avoid modifying in place
                    selected_points = merge_ranges(selected_points_copy)

                # Update the color of the clicked point
                scatter.set_color(subset_colors)
                fig.canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event', on_click)

    # Custom legend elements
    custom_legend = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10), 
        Line2D([0], [0], marker='o', color='w', markerfacecolor='aqua', markersize=10),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10)   
    ]
    # Add legend to the plot
    ax.legend(
        custom_legend,
        ["Bad", "Probably Bad", "Probably Good", "Good"],  # Custom labels
        loc='lower left', title="Data Quality"
    )

    # Add labels and title
    if data_type == "PRES":
        plt.xlabel("Index")
        plt.ylabel('Pressure')
        plt.title(f"Flag Range Pressure Graph for Profile: {profile_num} on {from_julian_day(float(date)).date()}")
    else:
        plt.xlabel(data_type)
        plt.ylabel('Pressure')
        plt.title(f"Flag Range PRES v {data_type} for Profile: {profile_num} on {from_julian_day(float(date)).date()}")

    plt.show()

    return colors

def flag_TS_data_graphs(sal, temp, date, lons, lats, pres, profile_num, temp_adjusted_qc, psal_adjusted_qc, ax=None):
    """
    Interactive Temperature-Salinity (TS) QC flagging graph for a single profile.

    Points are color-coded by the combined TEMP+PSAL quality:
      green  = both good
      aqua   = salinity bad (prob bad or bad), temperature good
      violet = temperature bad (prob bad or bad), salinity good
      red    = both bad

    Clicking a point cycles it through the four states. Sigma-t density contours
    are plotted in the background. Can be embedded as a subplot (pass ax) or shown
    standalone.

    Parameters
    ----------
    sal : ndarray (n_levels,)
        Practical salinity for a single profile.
    temp : ndarray (n_levels,)
        In-situ temperature (°C) for a single profile.
    date : float
        Julian day of the profile (referenced to 1950-01-01).
    lons, lats : float
        Profile longitude and latitude (used for optional GSW conversions).
    pres : ndarray (n_levels,)
        Pressure for a single profile.
    profile_num : int
        Profile number, used in the plot title.
    temp_adjusted_qc : ndarray (n_levels,)
        QC flags for TEMP_ADJUSTED.
    psal_adjusted_qc : ndarray (n_levels,)
        QC flags for PSAL_ADJUSTED.
    ax : matplotlib.axes.Axes, optional
        Axis to plot on; if None, a new figure/axis is created.

    Returns
    -------
    selected_points : list of int
        Combined QC state per point: 1=both good, 2=temp bad, 3=sal bad, 4=both bad.
        NOTE: clicking is only enabled in standalone mode (ax=None).
    """
    print_multiplot = False
    sal_copy = copy.deepcopy(sal)
    temp_copy = copy.deepcopy(temp)
    
    # Get rid of data conversion for QC
    # if (not np.isnan(lats)) and (not np.isnan(lons)):
    #     sal_copy = gsw.conversions.SA_from_SP(sal, temp, lons, lats)
    #     temp_copy = gsw.conversions.CT_from_t(sal, temp, pres)
    # else:
    #     sal_copy = sal
    #     temp_copy = temp

    # Define salinity and temperature bounds for the contour plot
    smin = np.nanmin(sal_copy) - 1
    smax = np.nanmax(sal_copy) + 1
    tmin = np.nanmin(temp_copy) - 1
    tmax = np.nanmax(temp_copy) + 1

    # Calculate grid cells needed in x and y dimensions
    xdim = int(np.ceil((smax - smin) / 0.1))
    ydim = int(np.ceil((tmax - tmin)))
    dens = np.zeros((ydim, xdim))

    # Create temp and salt vectors for density
    ti = np.linspace(0, ydim, ydim) + tmin
    si = np.linspace(1, xdim, xdim) * 0.1 + smin

    # Fill grid with densities
    for j in range(ydim):
        for i in range(xdim):
            dens[j, i] = gsw.rho(si[i], ti[j], 0)

    # Subtract 1000 to convert to sigma-t
    dens = dens - 1000

    # init colors arr
    colors = []
    # init selected points arr
    selected_points = []
    for sal_qc, temp_qc in zip(psal_adjusted_qc, temp_adjusted_qc):
        if (sal_qc == 4 or sal_qc == 3) and (temp_qc == 4 or temp_qc == 3):
            colors.append('red')      # Both salinity and temperature are bad
            selected_points.append(4)
        elif sal_qc == 4 or sal_qc == 3:
            colors.append('aqua')   # Only salinity is bad
            selected_points.append(3)
        elif temp_qc == 4 or temp_qc == 3:
            colors.append('violet')   # Only temperature is bad
            selected_points.append(2)
        else:
            colors.append('green')    # Neither is bad
            selected_points.append(1) 

    # Plot data
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        print_multiplot = True

    CS = ax.contour(si, ti, dens, linestyles='dashed', colors='k')
    ax.clabel(CS, fontsize=12, inline=1, fmt='%.2f')
    # Plot TS line
    ts_line = ax.plot(sal_copy, temp_copy, color='dimgrey', linewidth=2, zorder=1)
    # Plot the salinity-temperature relationship as a scatter plot
    scatter_plt = ax.scatter(sal_copy, temp_copy, color=colors, s=20, alpha=0.9, zorder = 2)
  
    # Use mplcursors to display profile numbers on hover
    cursor = mplcursors.cursor(scatter_plt, hover=True)
    def annotate_hover(sel):
        s, t = sel.target
        sel.annotation.set_text(f"Temperature: {t:.2f} C\nSalinity: {s:.2f} PSU")
    cursor.connect("add", annotate_hover)

    # click event
    def on_click(event):
        # Tells function to use a nonlocal selected points 
        nonlocal selected_points
        for scatter, graph_colors in [(scatter_plt, colors)]:
            cont, ind = scatter.contains(event)
            if cont:
                clicked_idx = ind["ind"][0]  
                # Get org color of point
                org_color = graph_colors[clicked_idx]
                # Cycle through color options
                if org_color == 'red':
                    graph_colors[clicked_idx] = 'aqua'
                    selected_points[clicked_idx] = 3
                elif org_color == 'aqua':
                    graph_colors[clicked_idx] = 'violet'
                    selected_points[clicked_idx] = 2
                elif org_color == 'violet':
                    graph_colors[clicked_idx] = 'green'
                    selected_points[clicked_idx] = 1
                else: # color is green
                    graph_colors[clicked_idx] = 'red'
                    selected_points[clicked_idx] = 4
                
                # Update the color of the clicked point
                scatter.set_color(graph_colors)
                fig.canvas.draw_idle()
    
    if print_multiplot == False:
        fig.canvas.mpl_connect('button_press_event', on_click)

    # Set x and y limits, labels, and title
    ax.set_xlim([smin + 0.75, smax - 0.75])
    ax.set_ylim([tmin + 0.75, tmax - 0.75])

    # Custom legend elements
    custom_legend = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10),    # Both bad
        Line2D([0], [0], marker='o', color='w', markerfacecolor='aqua', markersize=10),   # Salinity bad
        Line2D([0], [0], marker='o', color='w', markerfacecolor='violet', markersize=10), # Temperature bad
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10)   # Good data
    ]
    # Add legend to the plot
    ax.legend(
        custom_legend,
        ["Both bad", "Salinity bad", "Temperature bad", "Good data"],  # Custom labels
        loc='lower left', title="Data Quality"
    )

    ax.set_xlabel('Salinity (PSU)')
    ax.set_ylabel('In-Situ Temperature (degC)')
 
    if print_multiplot == False:
        ax.set_title(f"TS Graph for Profile: {profile_num} on {from_julian_day(float(date)).date()}")
        plt.tight_layout()
        plt.show()

    return selected_points

def deep_section_var_all(pressure, dates, COMP_vars, float_num, deep_section_compare_var):
    """
    Plot a depth section showing variable evolution over time.

    X-axis is date, Y-axis is depth (pressure, inverted), and color fill shows
    the variable value (TEMP or PSAL). Black contour lines are overlaid.

    Parameters
    ----------
    pressure : ndarray (n_profiles, n_levels)
        Pressure values for all profiles.
    dates : ndarray (n_profiles,)
        Julian days for each profile, referenced to 1950-01-01.
    COMP_vars : ndarray (n_profiles, n_levels)
        Variable values (TEMP or PSAL) for all profiles.
    float_num : str
        Float identifier used in the plot title.
    deep_section_compare_var : str
        Either 'TEMP' or 'PSAL' — sets the colorbar label.
    """
    # Create a meshgrid for dates and pressure, matching the temperature data
    X, Y = np.meshgrid(dates, np.linspace(np.nanmin(pressure), np.nanmax(pressure), pressure.shape[1]), indexing='ij')

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create plots
    contour = ax.pcolormesh(X, Y, COMP_vars, cmap='jet')
    ax.contour(X, Y, COMP_vars, colors = 'black')

    # Invert the y-axis to have depth increasing downwards
    ax.invert_yaxis()

    # Convert julian times to datetime objs and set tick labels
    date_formatter = DateFormatter('%Y-%m-%d')
    ticks =[datetime.datetime(1950, 1, 1) + (datetime.timedelta(days=float(juld)))  for juld in ax.get_xticks()]
    ax.xaxis.set_major_formatter(date_formatter)
    ax.xaxis.set_ticklabels([dt.strftime('%Y-%m-%d') for dt in ticks])

    # Add a colorbar
    cbar = plt.colorbar(contour, ax=ax)
    if deep_section_compare_var == "TEMP":
        cbar.set_label('Temperature In-Situ (°C)')
    elif deep_section_compare_var == "PSAL":
        cbar.set_label('Practical Salinity (psu)')
    else:
        raise Exception("deep_section_compare_var must be TEMP or PSAL not {}".format(deep_section_compare_var))

    # Set labels and title
    ax.set_xlabel('Date')
    ax.set_ylabel('Index')
    ax.set_title(f'Argo Float {float_num} Deep Section {deep_section_compare_var} Graph')

    plt.show()

def TS_graph_single_dataset_all_profile(df_SALs, df_TEMPs, df_JULD, df_LONs, df_LATs, df_PRESs, df_prof_nums, float_name):
    """
    Interactive TS diagram for a single float showing all profiles.

    Each profile is drawn as a line segment in a LineCollection colored by date
    (jet colormap). Sigma-t density contours are plotted in the background.
    Hovering shows the profile number; clicking toggles the profile in/out of the
    returned selection set.

    Parameters
    ----------
    df_SALs, df_TEMPs : ndarray (n_profiles, n_levels)
        Salinity and temperature arrays for all profiles.
    df_JULD : ndarray (n_profiles,)
        Julian days for each profile, referenced to 1950-01-01.
    df_LONs, df_LATs : ndarray (n_profiles,)
        Longitude and latitude for each profile.
    df_PRESs : ndarray (n_profiles, n_levels)
        Pressure (currently unused; reserved for SA/CT conversions).
    df_prof_nums : ndarray (n_profiles,)
        Profile numbers.
    float_name : str
        Float identifier used in the plot title.

    Returns
    -------
    selected_profiles : set of int
        Profile numbers clicked/selected by the user before closing the window.
    """
    selected_profiles = set()
    segment_index = None

    df_SALs_copy = copy.deepcopy(df_SALs)
    df_TEMPs_copy = copy.deepcopy(df_TEMPs)
    
    # Get rid of data conversions for QC
    # for i in np.arange(df_SALs.shape[0]):    # number of profiles
    #     df_SALs_copy[i, :] = gsw.conversions.SA_from_SP(df_SALs[i, :], df_TEMPs[i, :], df_LONs[i], df_LATs[i])
    #     df_TEMPs_copy[i, :] = gsw.conversions.CT_from_t(df_SALs[i, :], df_TEMPs[i, :], df_PRESs[i, :])
    
    # Define salinity and temperature bounds for the contour plot
    smin = np.nanmin(df_SALs_copy) - 1
    smax = np.nanmax(df_SALs_copy) + 1
    tmin = np.nanmin(df_TEMPs_copy) - 1
    tmax = np.nanmax(df_TEMPs_copy) + 1

    # Calculate grid cells needed in x and y dimensions
    xdim = int(np.ceil((smax - smin) / 0.1))
    ydim = int(np.ceil((tmax - tmin)))
    dens = np.zeros((ydim, xdim))

    # Create temp and salt vectors for density
    ti = np.linspace(0, ydim, ydim) + tmin
    si = np.linspace(1, xdim, xdim) * 0.1 + smin

    # Fill grid with densities
    for j in range(ydim):
        for i in range(xdim):
            dens[j, i] = gsw.rho(si[i], ti[j], 0)

    # Subtract 1000 to convert to sigma-t
    dens = dens - 1000

    ref_segments = [np.column_stack((df_SALs[i, :], df_TEMPs[i, :])) for i in range(df_SALs.shape[0])]
    ref_lc = LineCollection(ref_segments, array=df_JULD, cmap='jet', alpha=0.7)
    ref_lc.set_linewidth(1)

    # Attach the profile numbers to the LineCollection
    ref_lc.profile_numbers = df_prof_nums

    # Plot data
    fig, ax = plt.subplots(figsize=(10, 6))
    CS = plt.contour(si, ti, dens, linestyles='dashed', colors='k')
    plt.clabel(CS, fontsize=12, inline=1, fmt='%.2f')

    # Add LineCollection to plot
    ax = plt.gca()
    ax.add_collection(ref_lc)

    # Add colorbar with date
    cbar = plt.colorbar(ref_lc, ax=ax, label='Date')
    date_formatter = DateFormatter('%Y-%m-%d')

    # Convert colorbar ticks to regular dates
    cbar_ticks = [datetime.datetime(1950, 1, 1) + (datetime.timedelta(days=float(juld)))  for juld in cbar.get_ticks()]
    cbar.ax.yaxis.set_major_formatter(date_formatter)
    cbar.set_ticklabels([dt.strftime('%Y-%m-%d') for dt in cbar_ticks])

    # Set x and y limits, labels, and title
    plt.xlim([smin + 0.75, smax - 0.75])
    plt.ylim([tmin + 0.75, tmax - 0.75])
    plt.xlabel('Salinity (PSU)')
    plt.ylabel('In-Situ Temperature (degC)')
    plt.title(f"Argo Float {float_name} TS Graph")
    plt.tight_layout()

    # Use mplcursors to display profile numbers on hover
    cursor = mplcursors.cursor(ref_lc, hover=True)
    
    # Attach profile numbers to each segment
    def annotate_hover(sel):
        nonlocal segment_index
        segment_index = sel.index[0]  # Index of the segment in LineCollection
        profile_number = ref_lc.profile_numbers[segment_index]  # Get the corresponding profile number
        sel.annotation.set_text(f"Profile: {profile_number}")
    
    def on_click(event):
        nonlocal selected_profiles, segment_index
        if segment_index:
            profile_number = ref_lc.profile_numbers[segment_index]
            if profile_number in selected_profiles:
                selected_profiles.remove(profile_number)
                print(f"Removing {profile_number}")
            else:
                selected_profiles.add(profile_number)
                print(f"Adding {profile_number}")
            print(f"Selected Profiles: {selected_profiles}")
            print("=====================" + "====" * len(selected_profiles))
    fig.canvas.mpl_connect('button_press_event', on_click)

    cursor.connect("add", annotate_hover)
    plt.grid(True)
    plt.show()

    return selected_profiles

def del_bad_points(PRES_ADJUSTED_QC, TEMP_ADJUSTED_QC, PSAL_ADJUSTED_QC,
                   PSAL_ADJUSTED, TEMP_ADJUSTED, PRES_ADJUSTED):
    """
    Replace all points flagged as bad (QC=4) in any array with NaN across all three arrays.

    A combined mask is built: if PRES, TEMP, or PSAL is bad at a given level,
    that level is set to NaN in all three data arrays. This ensures consistency
    across paired variables before graphing or analysis.

    Parameters
    ----------
    PRES_ADJUSTED_QC, TEMP_ADJUSTED_QC, PSAL_ADJUSTED_QC : ndarray
        QC flag arrays. Values of 4 indicate bad data.
    PSAL_ADJUSTED, TEMP_ADJUSTED, PRES_ADJUSTED : ndarray
        Corresponding data arrays to be masked.

    Returns
    -------
    PSAL_ADJUSTED, TEMP_ADJUSTED, PRES_ADJUSTED : ndarray
        Data arrays with bad points replaced by NaN.
    """
    # Identify bad values (marked as 4) in each QC array
    pres_bad = (PRES_ADJUSTED_QC == 4)
    temp_bad = (TEMP_ADJUSTED_QC == 4)
    psal_bad = (PSAL_ADJUSTED_QC == 4)
    
    # Combine masks: mark as bad if any of the arrays have bad values
    combined_bad_mask = pres_bad | temp_bad | psal_bad
    
    # Replace bad values with NaN, preserving the original shape
    PSAL_ADJUSTED = np.where(combined_bad_mask, np.nan, PSAL_ADJUSTED)
    TEMP_ADJUSTED = np.where(combined_bad_mask, np.nan, TEMP_ADJUSTED)
    PRES_ADJUSTED = np.where(combined_bad_mask, np.nan, PRES_ADJUSTED)
    
    return PSAL_ADJUSTED, TEMP_ADJUSTED, PRES_ADJUSTED

## These functions are for the Cell Thermal Mass correction to determine if they were needed, obsolete now 
## =================================================================================================
def single_prof_datasnapshot(profile_num, argo_data, PSAL_ADJUSTED_Padj_CTM):
    """
    Display a 2x2 interactive QC panel for a single profile.

    Layout:
      Top-left:     flag_point_data_graphs for TEMP (click to flag individual points)
      Top-right:    flag_point_data_graphs for PSAL (click to flag individual points)
      Bottom-left:  flag_TS_data_graphs (TS diagram, view only in multi-panel mode)
      Bottom-right: text summary (profile number, datetime, lat/lon, instructions)

    Note: interactive clicking is only active in the top two panels (standalone axes).
    The TS subplot is display-only when embedded here.

    Parameters
    ----------
    profile_num : int
        Profile number to display.
    argo_data : dict
        Full intermediate netCDF data dict (from read_intermediate_nc_file).
    PSAL_ADJUSTED_Padj_CTM : ndarray (n_profiles, n_levels)
        CTM-corrected salinity to use in place of raw PSAL_ADJUSTED.
    """
    i = np.where(argo_data["PROFILE_NUMS"] == profile_num)[0][0]

    sal_arr = np.squeeze(PSAL_ADJUSTED_Padj_CTM[i])
    temp_arr = argo_data["TEMP_ADJUSTED"][i]
    pres_arr = argo_data["PRES_ADJUSTED"][i]
    temp_qc = argo_data["TEMP_ADJUSTED_QC"][i]
    psal_qc = argo_data["PSAL_ADJUSTED_QC"][i]
    lon = argo_data["LONs"][i]
    lat = argo_data["LATs"][i]
    juld = argo_data["JULDs"][i]

    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    # Populate the top-left subplot
    selected_points_temp = flag_point_data_graphs(temp_arr, pres_arr, "TEMP", temp_qc, argo_data["PROFILE_NUMS"][i], juld, ax=axs[0, 0], figure=fig)

    # Populate the top-right subplot
    selected_points_psal = flag_point_data_graphs(sal_arr, pres_arr, "PSAL", psal_qc, argo_data["PROFILE_NUMS"][i], juld, ax=axs[0, 1], figure=fig)

    # Populate the bottom-left subplot
    flag_TS_data_graphs(sal_arr, temp_arr, juld, lon, lat, pres_arr, profile_num, temp_qc, psal_qc, ax=axs[1, 0])

    # Fill bottom-right subplot with text
    timestamp = from_julian_day(float(juld))
    axs[1, 1].text(0.5, 0.7, f'Data Snapshot of Profile: {profile_num}', fontsize=12, ha='center', va='center')
    axs[1, 1].text(0.5, 0.5, f'Datetime of Profile: {timestamp.date()} {timestamp.strftime('%H:%M:%S')}', fontsize=12, ha='center', va='center')
    axs[1, 1].text(0.5, 0.3, f'Lat: {lat:.2f} Lon: {lon:.2f}', fontsize=12, ha='center', va='center')
    axs[1, 1].text(0.5, 0.1, f'Flag QC-point feature enabled for TEMP + PSAL graphs', fontsize=12, ha='center', va='center')
    axs[1, 1].axis('off')

    axs[1,0].grid(True)

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()
def compute_cellTM():
    """
    Compute the Cell Thermal Mass (CTM) corrected salinity for float F9186.

    For each profile:
      1. Extract the valid depth range (up to last non-NaN pressure level).
      2. Build an elapsed-time array from PTSCI_TIMESTAMPS (or cumulative NB_SAMPLE_CTD
         if use_timestamps=False).
      3. Mask out levels where sample count > 50 (bin averages unreliable).
      4. Call RBRargo3_celltm.RBRargo3_celltm() to get CTM-corrected temperature.
      5. Convert corrected temperature + original conductivity to practical salinity
         using GSW (SP_from_C).

    After computing PSAL_ADJUSTED_Padj_CTM, applies QC masks so that only
    levels with PSAL, PRES, and TEMP QC in [0,1,2] are kept.

    Then calls make_der_graph to display the correction results profile-by-profile.

    Hardcoded path: F9186_after_vi_new intermediate netCDF directory.
    """
    # Read in intermediate NETCDF files for QC arrs
    # fp = "C:\\Users\\szswe\\Desktop\\compare_floats_project\\data\\argo_to_nc\\F10051_after_visual_inspection"
    fp = "C:\\Users\\szswe\\Desktop\\compare_floats_project\\data\\csv_to_nc\\F9186_after_vi_new"
    float_num = "F9186_ADJUSTED_SAL"
    use_timestamps = True
    argo_data = read_intermediate_nc_file(fp)
 
    cell_tms = []
    for i in np.arange(len(argo_data["NB_SAMPLE_CTD"])):
        nan_index = np.where(~np.isnan(argo_data["PRESs"][i, :]))[0][-1] + 1
        
        if use_timestamps == True:
            # NO FIRST COUNT
            # TIMESTAMP and COUNTs arrays are equal
            times = []
            for ts in argo_data["PTSCI_TIMESTAMPS"][i, :nan_index]:
                times.append(datetime.datetime.strptime(str(ts)[:-2], "%Y%m%d%H%M%S"))
            times = np.array(times)
            time_deltas = []
            for dt in times:
                time_deltas.append(abs((dt - times[0]).total_seconds()))
            time_deltas = np.array(time_deltas)
            elptime = np.asarray(time_deltas)
        else:
            # KEEP FIRST COUNT
            # less trailing points, gets rid of last sum to keep shape
            elptime = np.cumsum(argo_data["NB_SAMPLE_CTD"][i, :nan_index])
            elptime = np.insert(elptime, 0, 0)[:-1]
        
        # Take data arrays
        temperature = argo_data["TEMP_ADJUSTED"][i, :nan_index]
        pressure = argo_data["PRES_ADJUSTED"][i, :nan_index]
        temp_cndc = argo_data["TEMP_CNDCs"][i, :nan_index]

        # Get rid of bad points 
        # WHERE COUNTS > 50: PSAL_ADJUSTED_QC, CNDC_ADJUSTED_QC, TEMP_ADJUSTED_QC are set to 3
        # 1) Get rid of bad points on all levels where COUNTS > 50
        # NOTE: if we have a level of ALL NaN's the RBR function gets weird... 
        count_mask = np.where(argo_data["NB_SAMPLE_CTD"][i, :nan_index] > 50)[0]
        temperature[count_mask] = np.nan
        #pressure[count_mask] = np.nan
        #temp_cndc[count_mask] = np.nan
        # 2) Get rid of bad points where TEMP, PRES, TEMP_CNDC QC arr's == (3/4)
        #temp_mask = np.where(((argo_data["TEMP_ADJUSTED_QC"][i, :nan_index] == 3) | (argo_data["TEMP_ADJUSTED_QC"][i, :nan_index] == 4)))[0]
        # pres_mask = np.where(((argo_data["PRES_ADJUSTED_QC"][i, :nan_index] == 3) | (argo_data["PRES_ADJUSTED_QC"][i, :nan_index] == 4)))[0]
        # temp_cndc_mask = np.where(((argo_data["TEMP_CNDC_QC"][i, :nan_index] == 3) | (argo_data["TEMP_CNDC_QC"][i, :nan_index] == 4)))[0]
        #temperature[temp_mask] = np.nan
        # pressure[pres_mask] = np.nan
        # temp_cndc[temp_cndc_mask] = np.nan

        a = RBRargo3_celltm.RBRargo3_celltm(temperature, pressure, temp_cndc, elptime)
        cell_tms.append(a)

    cell_tms = np.squeeze(np.array(list(itertools.zip_longest(*cell_tms, fillvalue=np.nan))).T)
    PSAL_ADJUSTED_Padj_CTM = gsw.SP_from_C(argo_data["CNDCs"] * 10, cell_tms, argo_data["PRES_ADJUSTED"])

    # apply qc_arr to data
    mask_var = np.isin(argo_data["PSAL_ADJUSTED_QC"], [0, 1, 2])
    mask_pres = np.isin(argo_data["PRES_ADJUSTED_QC"], [0, 1, 2])
    mask_temp = np.isin(argo_data["TEMP_ADJUSTED_QC"], [0, 1, 2])

    pres = np.where(mask_pres, argo_data["PRES_ADJUSTED"], np.nan)
    psal = np.where(mask_var, PSAL_ADJUSTED_Padj_CTM, np.nan)
    temp = np.where(mask_temp, argo_data["TEMP_ADJUSTED"], np.nan)
    
    #pres_v_var_all(pres, psal, argo_data["JULDs"], argo_data["PROFILE_NUMS"], "PSAL", float_num)
    #TS_graph_single_dataset_all_profile(psal, temp, argo_data["JULDs"], argo_data["LONs"], argo_data["LATs"], pres, argo_data["PROFILE_NUMS"], float_num)
    #single_prof_datasnapshot(285, argo_data, PSAL_ADJUSTED_Padj_CTM)
    make_der_graph(argo_data, PSAL_ADJUSTED_Padj_CTM)
def make_der_graph(argo_data, PSAL_ADJUSTED_Padj_CTM):
    """
    Plot side-by-side salinity comparison and temperature gradient for each profile.

    For each profile, displays a two-panel figure:
      Left panel:  Original PSAL_ADJUSTED (red) vs CTM-corrected salinity (green)
                   vs pressure (depth increasing downward).
      Right panel: dT/dz (temperature gradient) vs pressure midpoints, with a
                   zero-line reference to visualize sign changes.

    Used to visually assess the impact of the Cell Thermal Mass correction.

    Parameters
    ----------
    argo_data : dict
        Full intermediate netCDF data dict (from read_intermediate_nc_file).
    PSAL_ADJUSTED_Padj_CTM : ndarray (n_profiles, n_levels)
        CTM-corrected salinity array from compute_cellTM().
    """
    for i in np.arange(len(argo_data["PROFILE_NUMS"])):

        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle(f'Profile: {argo_data["PROFILE_NUMS"][i]}')

        pres = argo_data["PRES_ADJUSTED"][i]
        psal_old = argo_data["PSAL_ADJUSTED"][i]
        psal_new = PSAL_ADJUSTED_Padj_CTM[i]

        # Plot salinity on left-side axis
        #ax1.scatter(psal_old, pres, s=1, c="red")
        ax1.plot(psal_old, pres, c="red")
        #ax1.scatter(psal_new, pres, s=1, c="green")
        ax1.plot(psal_new, pres, c="green")
        # Plot configs
        ax1.grid(True)
        ax1.yaxis.set_inverted(True)
        ax1.set_title("Adjusted v Original Salinity")
        ax1.set_xlabel("Salinity")
        ax1.set_ylabel("Pressure")
        ax1.legend(
            [Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=7),
             Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=7)],
            ["Org Salinity", "Adjusted Salinity"], bbox_to_anchor=(0.5, -0.07), ncol = 2, loc='upper center'
        )

        # Plot derivative of temp/ depth on right side
        dT_dz = np.diff(argo_data["TEMP_ADJUSTED"][i]) / np.diff(pres) # divide the diff of adjacent elements of arrs
        pressure_mid = (pres[:-1] + pres[1:]) / 2 # Find the midpoint of pressures
        ax2.plot(dT_dz, pressure_mid)
        # Plot configs
        ax2.yaxis.set_inverted(True)
        ax2.set_title("dT/dz Graph")
        ax2.set_xlabel("dT/dz")
        ax2.set_ylabel("Pressure Midpoints")
        ax2.grid(True)
        ax2.axvline(0, color='black', linestyle='--', linewidth=1) # reference line

        plt.show()
## =================================================================================================
def main():
    print("Main function is a placeholder for testing individual graphing functions. Call specific functions as needed.")

if __name__ == '__main__':
    main()