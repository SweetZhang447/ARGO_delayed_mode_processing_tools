from datetime import datetime
import glob
import itertools
import os
from pathlib import Path
import gsw
from mpl_toolkits.basemap import Basemap
from geopy import distance

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy import stats
from tools import from_julian_day, read_intermediate_nc_file, to_julian_day
import netCDF4 as nc4

def read_ORP_WOOD_ctd():
    fp1 = Path(r"C:\Users\szswe\Desktop\sal_drift\ORP_WOOD\060671_20250712_2329DUNDEE_downcast_data.csv")
    # F9443 twin
    fp2 = Path(r"C:\Users\szswe\Desktop\sal_drift\ORP_WOOD\060671_20250714_1029_downcast_data.csv")
    # F10051 twin
    fp3 = Path(r"C:\Users\szswe\Desktop\sal_drift\ORP_WOOD\060671_20250724_0420_downcast_data.csv")
    fp4 = Path(r"C:\Users\szswe\Desktop\sal_drift\ORP_WOOD\060671_20250727_1003_downcast_data.csv")
    
    fp1_data = pd.read_csv(fp1)
    fp2_data = pd.read_csv(fp2)
    fp3_data = pd.read_csv(fp3)
    fp4_data = pd.read_csv(fp4)

    # drop cols not needed
    fp1_data = fp1_data.drop(columns=['Turbidity_NTU', 'N2_1_per_s2', 'PAR', 'Timestamp'])
    fp2_data = fp2_data.drop(columns=['Turbidity_NTU', 'N2_1_per_s2', 'PAR', 'Timestamp'])
    fp3_data = fp3_data.drop(columns=['Turbidity_NTU', 'N2_1_per_s2', 'PAR', 'Salinity_Source', 'Timestamp'])
    fp4_data = fp4_data.drop(columns=['Turbidity_NTU', 'N2_1_per_s2', 'PAR', 'Salinity_Source', 'Timestamp'])
    # Convert to dictionary 
    data_dict_fp1 = fp1_data.to_dict(orient='list')
    data_dict_fp2 = fp2_data.to_dict(orient='list')
    data_dict_fp3 = fp3_data.to_dict(orient='list')
    data_dict_fp4 = fp4_data.to_dict(orient='list')

    julds = [to_julian_day(datetime.strptime("2025-07-12", "%Y-%m-%d")),
             to_julian_day(datetime.strptime("2025-07-14", "%Y-%m-%d")),
             to_julian_day(datetime.strptime("2025-07-24", "%Y-%m-%d")),
             to_julian_day(datetime.strptime("2025-07-27", "%Y-%m-%d"))]
    psal = [data_dict_fp1["Salinity_PSU"], data_dict_fp2["Salinity_PSU"], data_dict_fp3["Salinity_PSU"], data_dict_fp4["Salinity_PSU"]]
    temp = [data_dict_fp1["Temperature_C"], data_dict_fp2["Temperature_C"], data_dict_fp3["Temperature_C"], data_dict_fp4["Temperature_C"]]
    pres = [data_dict_fp1["Pressure_dbar"], data_dict_fp2["Pressure_dbar"], data_dict_fp3["Pressure_dbar"], data_dict_fp4["Pressure_dbar"],]
    
    CTD_data = {
            "PSALs": np.squeeze(np.array(list(itertools.zip_longest(*psal, fillvalue=np.nan))).T),
            "TEMPs": np.squeeze(np.array(list(itertools.zip_longest(*temp, fillvalue=np.nan))).T),
            "PRESs": np.squeeze(np.array(list(itertools.zip_longest(*pres, fillvalue=np.nan))).T),
            "JULDs": np.asarray(julds, dtype=np.float64)
        }
    
    return CTD_data
def read_AXCTD_F9443(bin_size):
    filepath = Path(r"C:\Users\szswe\Desktop\sal_drift\AXCTD_F9443")
    # ORDER: 306, 310, 309
    AXCTD_files = sorted(glob.glob(os.path.join(filepath, "*.edf")))

    lats = np.asarray([70.9866333, 70.8426333, 70.8996833], dtype=np.float64)
    lons = np.asarray([-53.6777333, -53.6464833, -52.7366667], dtype=np.float64)
    julds = np.asarray([to_julian_day(datetime.strptime("8/12/2021 13:24:22", "%m/%d/%Y %H:%M:%S")), 
                        to_julian_day(datetime.strptime("8/12/2021 13:27:56", "%m/%d/%Y %H:%M:%S")), 
                        to_julian_day(datetime.strptime("8/12/2021 13:41:37", "%m/%d/%Y %H:%M:%S"))], 
                        dtype=np.float64)
    all_depth, all_temp, all_cndc= [], [], []

    for AXCTD_file in AXCTD_files:

        before_data = True
        field9found = False
        after_data = False
        depth, temp, cndc, = [], [], [], 

        with open(AXCTD_file, mode='r') as file:
            for line in file:
                if after_data == True:
                    line = line.split('\t')
                    depth.append(float(line[2]))
                    temp.append(float(line[3]))
                    cndc.append(float(line[4]))
                if before_data == True:
                    line = line.split(' ')
                    if field9found == True:
                        after_data = True
                        before_data = False
                    if "Field9" in line:
                        field9found = True

        all_depth.append(depth)
        all_temp.append(temp)
        all_cndc.append(cndc)
    
    # First, convert depth to pressure
    all_pres = []
    for i in [0, 1, 2]:
        all_pres.append(gsw.p_from_z(-np.asarray(all_depth[i], dtype=np.float64), lats[i]))
    # bin avg data - PRES, TEMP, CNDC
    binned_pres, binned_temp, binned_cdnc, binned_psal = [], [], [], []
    for i in [0, 1, 2]:
        bin_edges = np.arange(np.nanmin(all_pres[i]), np.nanmax(all_pres[i]) + 2, bin_size)
        binned_pres.append(stats.binned_statistic(all_pres[i], all_pres[i], 'mean', bins=bin_edges).statistic)
        binned_temp.append(stats.binned_statistic(all_pres[i], all_temp[i], 'mean', bins=bin_edges).statistic)
        binned_cdnc.append(stats.binned_statistic(all_pres[i], all_cndc[i], 'mean', bins=bin_edges).statistic)
    
    # Recalc practical salinity 
    gsw.SP_from_C(binned_cdnc[i], binned_temp[i], binned_pres[i])
    for i in [0, 1, 2]:
        binned_psal.append(gsw.SP_from_C(binned_cdnc[i], binned_temp[i], binned_pres[i]))

    return {
        "PRESs": np.squeeze(np.array(list(itertools.zip_longest(*binned_pres, fillvalue=np.nan)))).T,
        "TEMPs": np.squeeze(np.array(list(itertools.zip_longest(*binned_temp, fillvalue=np.nan)))).T,
        "CNDCs": np.squeeze(np.array(list(itertools.zip_longest(*binned_cdnc, fillvalue=np.nan)))).T,
        "PSALs": np.squeeze(np.array(list(itertools.zip_longest(*binned_psal, fillvalue=np.nan)))).T,
        "JULDs": np.squeeze(julds)
    }
def read_2021_melville_ctd(bin_size):

    filepath = Path(r"C:\Users\szswe\Desktop\sal_drift\AXCTDs_2021_Melville")
    # ORDER: 430, 432, 435, 428, 452, 450, 442, 445
    AXCTD_files = sorted(glob.glob(os.path.join(filepath, "*.edf")))

    lats = np.asarray([75.853217, 75.792500, 75.666467, 75.994550, 74.544467, 74.603533, 75.184917, 74.913400], dtype=np.float64)
    lons = np.asarray([-65.664283, -64.172183, -62.395183, -62.764333, -59.880250, -61.205367, -61.197317, -60.490917], dtype=np.float64)
    julds = np.asarray([to_julian_day(datetime.strptime("8/31/2021 16:04:49", "%m/%d/%Y %H:%M:%S")), 
                        to_julian_day(datetime.strptime("8/12/2021 13:27:56", "%m/%d/%Y %H:%M:%S")), 
                        to_julian_day(datetime.strptime("8/12/2021 13:41:37", "%m/%d/%Y %H:%M:%S"))], 
                        dtype=np.float64)
    all_depth, all_temp, all_cndc= [], [], []

    for AXCTD_file in AXCTD_files:

        before_data = True
        field9found = False
        after_data = False
        depth, temp, cndc, = [], [], [], 

        with open(AXCTD_file, mode='r') as file:
            for line in file:
                if after_data == True:
                    line = line.split('\t')
                    depth.append(float(line[2]))
                    temp.append(float(line[3]))
                    cndc.append(float(line[4]))
                if before_data == True:
                    line = line.split(' ')
                    if field9found == True:
                        after_data = True
                        before_data = False
                    if "Field9" in line:
                        field9found = True

        all_depth.append(depth)
        all_temp.append(temp)
        all_cndc.append(cndc)
    
    # First, convert depth to pressure
    all_pres = []
    for i in [0, 1, 2]:
        all_pres.append(gsw.p_from_z(-np.asarray(all_depth[i], dtype=np.float64), lats[i]))
    # bin avg data - PRES, TEMP, CNDC
    binned_pres, binned_temp, binned_cdnc, binned_psal = [], [], [], []
    for i in [0, 1, 2]:
        bin_edges = np.arange(np.nanmin(all_pres[i]), np.nanmax(all_pres[i]) + 2, bin_size)
        binned_pres.append(stats.binned_statistic(all_pres[i], all_pres[i], 'mean', bins=bin_edges).statistic)
        binned_temp.append(stats.binned_statistic(all_pres[i], all_temp[i], 'mean', bins=bin_edges).statistic)
        binned_cdnc.append(stats.binned_statistic(all_pres[i], all_cndc[i], 'mean', bins=bin_edges).statistic)
    
    # Recalc practical salinity 
    gsw.SP_from_C(binned_cdnc[i], binned_temp[i], binned_pres[i])
    for i in [0, 1, 2]:
        binned_psal.append(gsw.SP_from_C(binned_cdnc[i], binned_temp[i], binned_pres[i]))

    return {
        "PRESs": np.squeeze(np.array(list(itertools.zip_longest(*binned_pres, fillvalue=np.nan)))).T,
        "TEMPs": np.squeeze(np.array(list(itertools.zip_longest(*binned_temp, fillvalue=np.nan)))).T,
        "CNDCs": np.squeeze(np.array(list(itertools.zip_longest(*binned_cdnc, fillvalue=np.nan)))).T,
        "PSALs": np.squeeze(np.array(list(itertools.zip_longest(*binned_psal, fillvalue=np.nan)))).T,
        "JULDs": np.squeeze(julds)
    }

def gen_comparison_graph():
    
    ascent_fp = Path(r"C:\Users\szswe\Desktop\DMODE_processing\all_data_files\F10051\F10051_FTR_ascent")
    bin_avg_descent_fp = Path(r"C:\Users\szswe\Desktop\DMODE_processing\all_data_files\F10051\F10051_FTR")
    
    ascent_dt = read_intermediate_nc_file(ascent_fp)
    descent_bin_avg_dt = read_intermediate_nc_file(bin_avg_descent_fp)

    # make salinity graph ontop of each other
    fig, ax = plt.subplots()

    profile_108_ascent = np.where(ascent_dt["PROFILE_NUMS"] == 108)[0][0]
    profile_109_descent = np.where(descent_bin_avg_dt["PROFILE_NUMS"] == 109)[0][0]
    
    plt.plot(ascent_dt["PSALs"][profile_108_ascent], ascent_dt["PRESs"][profile_108_ascent], color = "red")
    plt.plot(descent_bin_avg_dt["PSALs"][profile_109_descent], descent_bin_avg_dt["PRESs"][profile_109_descent], color = "blue")

    # Invert y-axis and add grid
    plt.gca().invert_yaxis()
    plt.grid(visible=True)
    plt.xlabel("Salinity")
    plt.ylabel("Pressure")

    custom_legend = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10),
                ]
    # Add legend to the plot
    ax.legend(
        custom_legend,
        ["Profile 108 Ascent Data", "Profile 109 Bin-avg Descent Data"],  # Custom labels
        loc='lower left', title="Data Quality"
    )
    plt.title(f"F10051 Profile Comparison")

    plt.show()
def position_graph(lat_pt, lon_pt, df_LATs, df_LONs, prof_num, float_name, sort_by_distance):

    df_LATs = df_LATs.flatten()
    df_LONs = df_LONs.flatten()

    # FEATURE: sort by distance
    if sort_by_distance > 0:
        coords_pts = list(zip(lat_pt, lon_pt))[0]
        coords = list(zip(df_LATs, df_LONs))

        for i in range(len(coords)):
            if not np.isnan(coords[i]).any():
                if distance.distance(coords_pts, coords[i]).miles > sort_by_distance:
                    df_LONs[i] = np.NaN
                    df_LATs[i] = np.NaN

    # Make graph
    fig, ax = plt.subplots(figsize=(9, 9))

    m = Basemap(projection='gnom', lat_0=72, lon_0=-38,
                width=5000000, height=4500000, resolution='h', ax=ax)
    m.fillcontinents(color="#FFDDCC")
    m.drawmapboundary(fill_color="#DDEEFF")
    m.drawcoastlines()
    ax.set_title(f"{float_name}")

    x, y = m(df_LONs, df_LATs)
    m.scatter(x, y, marker='o', color='r')
 
    # graph F9186
    x, y = m(lon_pt, lat_pt)
    m.scatter(x, y, marker='o', color='g', s=50)
    # ---- Label each profile with its number ----
    for i, xx, yy in zip(prof_num, x, y):
        if i == 364:
            ax.text(xx, yy, str(i), fontsize=15, color='black', ha='left', va='bottom')
        else:
            ax.text(xx, yy, str(i), fontsize=8, color='black', ha='left', va='bottom')

    # draw distance circle
    if sort_by_distance > 0:
        # 100: # of points to use for circle's perimeter
        m.tissot(lon_pt[0], lat_pt[0],  sort_by_distance/ 69.0, 100, facecolor='none', edgecolor='blue', linestyle='--')

    """
    # includes low res code
    fig, ax = plt.subplots(1, 2, figsize=(12, 8))
    for i, res in enumerate(['l', 'h']):
        print(i)
        m = Basemap(projection='gnom', lat_0=72, lon_0=-38,
                    width=2700000, height=3400000, resolution='h', ax=ax[i])
        m.fillcontinents(color="#FFDDCC")
        m.drawmapboundary(fill_color="#DDEEFF")
        m.drawcoastlines()
        ax[i].set_title("resolution='{0}'".format(res))

        x, y = m(df_LONs, df_LATs)
        m.scatter(x, y, marker='o', color='r')

        # graph our float point
        x, y = m(lon_pt, lat_pt)
        m.scatter(x, y, marker='o', color='g', s=50)

        # draw distance circle
        if sort_by_distance > 0:
            # 100: # of points to use for circle's perimeter
            m.tissot(lon_pt[0], lat_pt[0],  sort_by_distance/ 69.0, 100, facecolor='none', edgecolor='blue', linestyle='--')

    """

    plt.show()
def anotha_comparison_graph():

    CTD_data = read_ORP_WOOD_ctd()
    bin_avg_descent_fp = Path(r"C:\Users\szswe\Desktop\DMODE_processing\all_data_files\F10051\F10051_VI")
    F9186_new_data = Path(r"C:\Users\szswe\Desktop\DMODE_processing\all_data_files\F9186\F9186_new_dmode\FTR")
    F10051_data = read_intermediate_nc_file(F9186_new_data)

    # find nearest F10051 profile to CTD
    # Find index of nearest value
    idx = np.argmin(np.abs(F10051_data["JULDs"] - CTD_data["JULDs"][2]))
    print(f"F10051 Nearest Date: {from_julian_day(np.float64(F10051_data["JULDs"][idx]))}")
    print(f"F10051 Profile Number: {F10051_data["PROFILE_NUMS"][idx]}")
    print(f"CTD Date: {from_julian_day(CTD_data["JULDs"][2])}")

    # Graph Nicole's CTD with nearest F10051
    # make salinity graph ontop of each other
    fig, ax = plt.subplots()

    plt.plot(CTD_data["PSALs"][2], CTD_data["PRESs"][2], color = "red")
    plt.plot(F10051_data["PSALs"][idx], F10051_data["PRESs"][idx], color = "blue")

    # Invert y-axis and add grid
    plt.gca().invert_yaxis()
    plt.grid(visible=True)
    plt.xlabel("Salinity")
    plt.ylabel("Pressure")

    custom_legend = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10),
                ]
    # Add legend to the plot
    ax.legend(
        custom_legend,
        ["Nicole CTD", "F9186_new"],  # Custom labels
        loc='lower left', title="Data Quality"
    )
    plt.title(f"F9186 v Nicole CTD")

    plt.show()

    # Add "Avg PSAL 800-900m text for both floats"
    pres_filter_f10051 = np.where((F10051_data["PRESs"][idx] >= 800) & (F10051_data["PRESs"][idx] <= 900))
    pres_filter_ctd = np.where((CTD_data["PRESs"][2] >= 800) & (CTD_data["PRESs"][2] <= 900))
    print(f"AVG PSAL 800-900m")
    print(f"CTD: {np.average(CTD_data["PSALs"][2][pres_filter_ctd])}")
    print(f"F9186: {np.average(F10051_data["PSALs"][idx][pres_filter_f10051])}")

def uhm_forgot():
        
    # Avg PSAL graph
    F9186_fp = Path(r"C:\Users\szswe\Desktop\DMODE_processing\all_data_files\F9186\F9186_new_dmode\FTRVI")
    F10051_fp = Path(r"C:\Users\szswe\Desktop\DMODE_processing\all_data_files\F10051\F10051_VI") 
    
    F9186_df = read_intermediate_nc_file(F9186_fp)
    F10051_df = read_intermediate_nc_file(F10051_fp)

    # apply QC
    F9186_df["PSALs"][np.where((F9186_df["PSAL_ADJUSTED_QC"] == 3) | (F9186_df["PSAL_ADJUSTED_QC"] == 4))] = np.nan
    F9186_df["PRESs"][np.where((F9186_df["PRES_ADJUSTED_QC"] == 3) | (F9186_df["PRES_ADJUSTED_QC"] == 4))] = np.nan
    F10051_df["PSALs"][np.where((F10051_df["PSAL_ADJUSTED_QC"] == 3) | (F10051_df["PSAL_ADJUSTED_QC"] == 4))] = np.nan
    F10051_df["PRESs"][np.where((F10051_df["PRES_ADJUSTED_QC"] == 3) | (F10051_df["PRES_ADJUSTED_QC"] == 4))] = np.nan
    F9186_df["TEMPs"][np.where((F9186_df["TEMP_ADJUSTED_QC"] == 3) | (F9186_df["TEMP_ADJUSTED_QC"] == 4))] = np.nan
    F10051_df["TEMPs"][np.where((F10051_df["TEMP_ADJUSTED_QC"] == 3) | (F10051_df["TEMP_ADJUSTED_QC"] == 4))] = np.nan
    
    #apply psal offset on F9186
    F9186_df["PSALs"] = F9186_df["PSALs"] + 0.025

    # filter for overlapping dates F9186
    F9186_date_filter_1d = F9186_df["JULDs"] >= F10051_df["JULDs"][0]
    F9186_date_filter = F9186_date_filter_1d[:, np.newaxis]
    F9186_pres = np.where(F9186_date_filter, F9186_df["PRESs"], np.nan)
    F9186_psal = np.where(F9186_date_filter, F9186_df["PSALs"], np.nan)
    F9186_temp  = np.where(F9186_date_filter, F9186_df["TEMPs"], np.nan)
    
    F9186_df["LATs"] = np.where(F9186_date_filter_1d, F9186_df["LATs"], np.nan)
    F9186_df["LONs"] = np.where(F9186_date_filter_1d, F9186_df["LONs"], np.nan)
    
    # prof_num = np.where(F9186_date_filter_1d, F9186_df["PROFILE_NUMS"], np.nan)
    # #prof_num = F10051_df["PROFILE_NUMS"]
    # position_graph(F9186_df["LATs"], F9186_df["LONs"], F10051_df["LATs"], F10051_df["LONs"], prof_num, "F10051 (red) v F9186 (green)", 0)
    # raise Exception

    # filter pres
    pres_min = 700
    pres_max = 800
    
    F10051_pres_filter = (F10051_df["PRESs"]>= pres_min) & (F10051_df["PRESs"] <= pres_max)
    F10051_psal = np.where(F10051_pres_filter, F10051_df["PSALs"], np.nan)
    #F10051_psal = np.where(F10051_pres_filter, F10051_df["TEMPs"], np.nan)
    F9186_pres_filter = (F9186_pres >= pres_min) & (F9186_pres <= pres_max)
    F9186_psal = np.where(F9186_pres_filter, F9186_psal, np.nan)
    #F9186_psal = np.where(F9186_pres_filter, F9186_temp, np.nan)
    # find avg psal
    F10051_avg = np.nanmean(F10051_psal, axis = 1)
    F9186_avg = np.nanmean(F9186_psal, axis = 1)
  
    # plot
    fig, ax = plt.subplots()

    F10051_df["JULDs"] = np.array([from_julian_day(j) for j in F10051_df["JULDs"]])
    F9186_df["JULDs"] = np.array([from_julian_day(j) for j in F9186_df["JULDs"]])
    plt.scatter(F10051_df["JULDs"], F10051_avg, color = "red")
    plt.plot(F10051_df["JULDs"], F10051_avg, color = "red")
    plt.scatter(F9186_df["JULDs"], F9186_avg, color = "blue")
    plt.plot(F9186_df["JULDs"], F9186_avg, color = "blue")

    plt.grid(visible=True)
    plt.xlabel("Date")
    plt.ylabel("Temp")
    custom_legend = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10),
                ]
    # Add legend to the plot
    ax.legend(
        custom_legend,
        ["F10051", "F9186"],  # Custom labels
        loc='lower left', title="Data Quality"
    )
    plt.title(f"Avg Temp F10051 and F9186 depth range {pres_min}-{pres_max}")

    plt.show()

def read_2903449():
    nc_filepath = Path(r"C:\Users\szswe\Desktop\sal_drift\2903449")
    nc_files = glob.glob(os.path.join(nc_filepath, "*.nc"))
    PRESs = []
    TEMPs = []
    PSALs = []
    JULDs = []
    PROF_NUMs = []

    for file in nc_files:
        nc = nc4.Dataset(file)
        PRESs.append(np.squeeze(nc.variables['PRES'][:].filled(np.NaN)))
        TEMPs.append(np.squeeze(nc.variables['TEMP'][:].filled(np.NaN)))
        PSALs.append(np.squeeze(nc.variables['PSAL'][:].filled(np.NaN)))
        JULDs.append(nc.variables['JULD'][:].filled(np.NaN))
        PROF_NUMs.append(nc.variables['CYCLE_NUMBER'][:].filled(np.NaN))
    return{
        "PRESs": np.squeeze(np.array(list(itertools.zip_longest(*PRESs, fillvalue=np.nan)))).T,
        "TEMPs": np.squeeze(np.array(list(itertools.zip_longest(*TEMPs, fillvalue=np.nan)))).T,
        "PSALs": np.squeeze(np.array(list(itertools.zip_longest(*PSALs, fillvalue=np.nan)))).T,
        "JULDs": np.squeeze(JULDs),
        "PROF_NUMs": np.squeeze(PROF_NUMs)
    }

def F9443_graph():
    # Read in F9443 data
    nc_filepath = Path(r"C:\Users\szswe\Desktop\DMODE_processing\all_data_files\F9443\F9443_VI")
    F9443_data = read_intermediate_nc_file(nc_filepath)
    F9443_data["PSALs"][np.where((F9443_data["PSAL_ADJUSTED_QC"] == 3) | (F9443_data["PSAL_ADJUSTED_QC"] == 4))] = np.nan
    F9443_data["PRESs"][np.where((F9443_data["PRES_ADJUSTED_QC"] == 3) | (F9443_data["PRES_ADJUSTED_QC"] == 4))] = np.nan
   
    # Read in AXCTD
    AXCTD_9443 = read_AXCTD_F9443(1)
    # Read in Nicole CTD - index: 1 for all dict elements
    nicole_CTD = read_ORP_WOOD_ctd()
    # Read in new float profiles
    float_CTD = read_2903449()

    # Find closest F9443 profile close to CTDs
    print("AXCTD")
    idx_AXCTD = np.argmin(np.abs(F9443_data["JULDs"] - AXCTD_9443["JULDs"][1]))
    print(f"F9443 Nearest Date: {from_julian_day(np.float64(F9443_data["JULDs"][idx_AXCTD]))}")
    print(f"AXCTDs Date: {from_julian_day(np.float64(AXCTD_9443["JULDs"][1]))}")
    print(f"F9443 Nearest Profile: {F9443_data["PROFILE_NUMS"][idx_AXCTD]}")
    # INDEX: 1
    
    print("Nicole CTD")
    idx_NICOLE = np.argmin(np.abs(F9443_data["JULDs"] - nicole_CTD["JULDs"][1]))
    print(f"F9443 Nearest Date: {from_julian_day(np.float64(F9443_data["JULDs"][idx_NICOLE]))}")
    print(f"Nicole Date: {from_julian_day(np.float64(nicole_CTD["JULDs"][1]))}")
    print(f"F9443 Nearest Profile: {F9443_data["PROFILE_NUMS"][idx_NICOLE]}")

    # print("2903449 Profiles")
    # idx_FLOAT_CTD = F9443_data["JULDs"] >= float_CTD["JULDs"][0] 
    # F9443_date_filter = idx_FLOAT_CTD[:, np.newaxis]
    # F9443_pres = np.where(F9443_date_filter, float_CTD["PRESs"], np.nan)
    # F9443_psal = np.where(F9443_date_filter, float_CTD["PSALs"], np.nan)
    
    # Find avg PSAL at DEPTH
    pres_min = 500
    pres_max = 600
    # F9443
    F9443_pres_filter = (F9443_data["PRESs"]>= pres_min) & (F9443_data["PRESs"] <= pres_max)
    F9443_temp = np.where(F9443_pres_filter, F9443_data["TEMPs"], np.nan)
    F9443_psal = np.where(F9443_pres_filter, F9443_data["PSALs"], np.nan)
    # AXCTD
    AXCTD_pres_filter = (AXCTD_9443["PRESs"] >= pres_min) & (AXCTD_9443["PRESs"] <= pres_max)
    AXCTD_temp = np.where(AXCTD_pres_filter, AXCTD_9443["TEMPs"] , np.nan)
    AXCTD_psal = np.where(AXCTD_pres_filter, AXCTD_9443["PSALs"] , np.nan)
    # Nicole
    nicole_pres_filter = (nicole_CTD["PRESs"][1] >= pres_min) & (nicole_CTD["PRESs"][1] <= pres_max)
    nicole_temp = np.where(nicole_pres_filter, nicole_CTD["TEMPs"][1] , np.nan)
    nicole_psal = np.where(nicole_pres_filter, nicole_CTD["PSALs"][1] , np.nan)
    # Other float
    float_CTD_filter = (float_CTD["PRESs"] >= pres_min) & (float_CTD["PRESs"] <= pres_max)
    CTD_temp = np.where(float_CTD_filter, float_CTD["TEMPs"] , np.nan)
    CTD_psal = np.where(float_CTD_filter, float_CTD["PSALs"] , np.nan)
    # find avg psal
    F9443_avg = np.nanmean(F9443_psal, axis = 1)
    AXCTD_avg = np.nanmean(AXCTD_psal, axis = 1)
    nicole_avg = np.nanmean(nicole_psal)
    float_avg = np.nanmean(CTD_psal, axis=1)
    # find avg temp
    F9443_avg_t = np.nanmean(F9443_temp, axis = 1)
    AXCTD_avg_t = np.nanmean(AXCTD_temp, axis = 1)
    nicole_avg_t = np.nanmean(nicole_temp)
    float_avg_t = np.nanmean(CTD_temp, axis=1)

    # Generate graph - avg PSAL at depth
    fig, ax = plt.subplots()

    # F9443_data["JULDs"] = np.array([from_julian_day(j) for j in F9443_data["JULDs"]])
    # plt.scatter(F9443_data["JULDs"], F9443_avg, color = "red")
    # plt.plot(F9443_data["JULDs"], F9443_avg, color = "red")
    
    # AXCTD_9443["JULDs"] = np.array([from_julian_day(j) for j in AXCTD_9443["JULDs"]])
    # plt.scatter(AXCTD_9443["JULDs"], AXCTD_avg, color = "blue")
    # plt.plot(AXCTD_9443["JULDs"], AXCTD_avg, color = "blue")
    
    # float_CTD["JULDs"] = np.array([from_julian_day(j) for j in float_CTD["JULDs"]])
    # plt.scatter(float_CTD["JULDs"], float_avg, color = "purple")
    # plt.plot(float_CTD["JULDs"], float_avg, color = "purple")

    # plt.scatter(from_julian_day(nicole_CTD["JULDs"][1]), nicole_avg, color = "green")

    plt.scatter(F9443_avg, F9443_avg_t, color = "red")
    plt.plot(F9443_avg, F9443_avg_t, color = "red")
    
    plt.scatter(AXCTD_avg, AXCTD_avg_t, color = "blue")
    plt.plot(AXCTD_avg, AXCTD_avg_t, color = "blue")
    
    plt.scatter(float_avg, float_avg_t, color = "purple")
    plt.plot(float_avg, float_avg_t, color = "purple")

    plt.scatter(nicole_avg, nicole_avg_t, color = "green")

    plt.grid(visible=True)
    plt.xlabel("Date")
    plt.ylabel("Temp")
    custom_legend = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=10),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10),
                ]
    # Add legend to the plot
    ax.legend(
        custom_legend,
        ["F9443", "AXCTD", "290344", "Nicole"],  # Custom labels
        loc='lower left', title="Data Quality"
    )
    plt.title(f"Avg PSAL depth range {pres_min}-{pres_max}")

    plt.show()

def main():
    # Read in F9444 data
    nc_filepath = Path(r"C:\Users\szswe\Desktop\DMODE_processing\all_data_files\F9444\F9444_VI")
    F9444_data = read_intermediate_nc_file(nc_filepath)
    # Read in F9185 data
    nc_filepath = Path(r"C:\Users\szswe\Desktop\DMODE_processing\all_data_files\F9185\F9185_VI")
    F9185_data = read_intermediate_nc_file(nc_filepath)
    # Apply QC to floats - PSAL
    F9444_data["PSALs"][np.where((F9444_data["PSAL_ADJUSTED_QC"] == 3) | (F9444_data["PSAL_ADJUSTED_QC"] == 4))] = np.nan
    F9444_data["PRESs"][np.where((F9444_data["PRES_ADJUSTED_QC"] == 3) | (F9444_data["PRES_ADJUSTED_QC"] == 4))] = np.nan
    F9185_data["PSALs"][np.where((F9185_data["PSAL_ADJUSTED_QC"] == 3) | (F9185_data["PSAL_ADJUSTED_QC"] == 4))] = np.nan
    F9185_data["PRESs"][np.where((F9185_data["PRES_ADJUSTED_QC"] == 3) | (F9185_data["PRES_ADJUSTED_QC"] == 4))] = np.nan

    # Read in AXCTD
    AXCTD_9443 = read_AXCTD_F9443(1)
    # Read in Nicole CTD - index: 1 for all dict elements
    nicole_CTD = read_ORP_WOOD_ctd()
    # Read in new float profiles
    float_CTD = read_2903449()

    # Find closest F9443 profile close to CTDs
    print("AXCTD")
    idx_AXCTD = np.argmin(np.abs(F9443_data["JULDs"] - AXCTD_9443["JULDs"][1]))
    print(f"F9443 Nearest Date: {from_julian_day(np.float64(F9443_data["JULDs"][idx_AXCTD]))}")
    print(f"AXCTDs Date: {from_julian_day(np.float64(AXCTD_9443["JULDs"][1]))}")
    print(f"F9443 Nearest Profile: {F9443_data["PROFILE_NUMS"][idx_AXCTD]}")
    # INDEX: 1
    
    print("Nicole CTD")
    idx_NICOLE = np.argmin(np.abs(F9443_data["JULDs"] - nicole_CTD["JULDs"][1]))
    print(f"F9443 Nearest Date: {from_julian_day(np.float64(F9443_data["JULDs"][idx_NICOLE]))}")
    print(f"Nicole Date: {from_julian_day(np.float64(nicole_CTD["JULDs"][1]))}")
    print(f"F9443 Nearest Profile: {F9443_data["PROFILE_NUMS"][idx_NICOLE]}")

    # print("2903449 Profiles")
    # idx_FLOAT_CTD = F9443_data["JULDs"] >= float_CTD["JULDs"][0] 
    # F9443_date_filter = idx_FLOAT_CTD[:, np.newaxis]
    # F9443_pres = np.where(F9443_date_filter, float_CTD["PRESs"], np.nan)
    # F9443_psal = np.where(F9443_date_filter, float_CTD["PSALs"], np.nan)
    
    # Find avg PSAL at DEPTH
    pres_min = 500
    pres_max = 600
    # F9443
    F9443_pres_filter = (F9443_data["PRESs"]>= pres_min) & (F9443_data["PRESs"] <= pres_max)
    F9443_temp = np.where(F9443_pres_filter, F9443_data["TEMPs"], np.nan)
    F9443_psal = np.where(F9443_pres_filter, F9443_data["PSALs"], np.nan)
    # AXCTD
    AXCTD_pres_filter = (AXCTD_9443["PRESs"] >= pres_min) & (AXCTD_9443["PRESs"] <= pres_max)
    AXCTD_temp = np.where(AXCTD_pres_filter, AXCTD_9443["TEMPs"] , np.nan)
    AXCTD_psal = np.where(AXCTD_pres_filter, AXCTD_9443["PSALs"] , np.nan)
    # Nicole
    nicole_pres_filter = (nicole_CTD["PRESs"][1] >= pres_min) & (nicole_CTD["PRESs"][1] <= pres_max)
    nicole_temp = np.where(nicole_pres_filter, nicole_CTD["TEMPs"][1] , np.nan)
    nicole_psal = np.where(nicole_pres_filter, nicole_CTD["PSALs"][1] , np.nan)
    # Other float
    float_CTD_filter = (float_CTD["PRESs"] >= pres_min) & (float_CTD["PRESs"] <= pres_max)
    CTD_temp = np.where(float_CTD_filter, float_CTD["TEMPs"] , np.nan)
    CTD_psal = np.where(float_CTD_filter, float_CTD["PSALs"] , np.nan)
    # find avg psal
    F9443_avg = np.nanmean(F9443_psal, axis = 1)
    AXCTD_avg = np.nanmean(AXCTD_psal, axis = 1)
    nicole_avg = np.nanmean(nicole_psal)
    float_avg = np.nanmean(CTD_psal, axis=1)
    # find avg temp
    F9443_avg_t = np.nanmean(F9443_temp, axis = 1)
    AXCTD_avg_t = np.nanmean(AXCTD_temp, axis = 1)
    nicole_avg_t = np.nanmean(nicole_temp)
    float_avg_t = np.nanmean(CTD_temp, axis=1)

    # Generate graph - avg PSAL at depth
    fig, ax = plt.subplots()

    # F9443_data["JULDs"] = np.array([from_julian_day(j) for j in F9443_data["JULDs"]])
    # plt.scatter(F9443_data["JULDs"], F9443_avg, color = "red")
    # plt.plot(F9443_data["JULDs"], F9443_avg, color = "red")
    
    # AXCTD_9443["JULDs"] = np.array([from_julian_day(j) for j in AXCTD_9443["JULDs"]])
    # plt.scatter(AXCTD_9443["JULDs"], AXCTD_avg, color = "blue")
    # plt.plot(AXCTD_9443["JULDs"], AXCTD_avg, color = "blue")
    
    # float_CTD["JULDs"] = np.array([from_julian_day(j) for j in float_CTD["JULDs"]])
    # plt.scatter(float_CTD["JULDs"], float_avg, color = "purple")
    # plt.plot(float_CTD["JULDs"], float_avg, color = "purple")

    # plt.scatter(from_julian_day(nicole_CTD["JULDs"][1]), nicole_avg, color = "green")

    plt.scatter(F9443_avg, F9443_avg_t, color = "red")
    plt.plot(F9443_avg, F9443_avg_t, color = "red")
    
    plt.scatter(AXCTD_avg, AXCTD_avg_t, color = "blue")
    plt.plot(AXCTD_avg, AXCTD_avg_t, color = "blue")
    
    plt.scatter(float_avg, float_avg_t, color = "purple")
    plt.plot(float_avg, float_avg_t, color = "purple")

    plt.scatter(nicole_avg, nicole_avg_t, color = "green")

    plt.grid(visible=True)
    plt.xlabel("Date")
    plt.ylabel("Temp")
    custom_legend = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=10),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10),
                ]
    # Add legend to the plot
    ax.legend(
        custom_legend,
        ["F9443", "AXCTD", "290344", "Nicole"],  # Custom labels
        loc='lower left', title="Data Quality"
    )
    plt.title(f"Avg PSAL depth range {pres_min}-{pres_max}")

    plt.show()


if __name__ == '__main__':
 
    main()