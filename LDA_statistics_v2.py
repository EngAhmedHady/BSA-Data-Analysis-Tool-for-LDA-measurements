# -*- coding: utf-8 -*-
"""
Created on Wed May 21 10:46:29 2025

@author: Ahmed H. Hanfy
"""
import os
import numpy as np
import pandas as pd
import xlwings as xw
from scipy import stats
import matplotlib.pyplot as plt

# Set matplotlib parameters globally for consistent plotting style
plt.rcParams.update({'font.size': 30})
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sens serif"
})

#%% Supporting functions
def histogram_title(points):
    """
    Generates a formatted title for a histogram based on a list of points.

    Parameters
    ----------
    points : list of str => A list of strings representing data points or file identifiers.

    Returns
    -------
    str => A formatted string suitable for a histogram title.
    """
    if not points:
        return ""
    elif len(points) == 1:
        return points[0]
    elif len(points) == 2:
        return " and ".join(points)
    else:
        return ", ".join(points[:-1]) + " and " + points[-1]
    
def filter_and_shift(df, group_cols, col_to_filter, lower_bound, upper_bound):
    """
    Filters out (shifts up) rows *only* in the specified group of columns
    where 'col_to_filter' is outside [lower_bound, upper_bound].

    Parameters
    ----------
    df : pd.DataFrame => DataFrame containing all columns.
    group_cols : list of str => The columns in the "group" to be shifted together, e.g.,
                                ["AT [ms]", "TT [µs]", "LDA1 [m/s]"].
    col_to_filter : str => The column in group_cols used to check the out-of-range condition,
                           e.g., "LDA1 [m/s]".
    lower_bound, upper_bound : float => Valid range for col_to_filter. Values outside this range
                                        will be removed (shifted up) in that group.

    Returns
    -------
    df : pd.DataFrame => The same DataFrame with the specified group shifted up for
                         out-of-range rows. Columns in the group that lost rows will
                         have NaN (or None) at the bottom.
    """

    # Convert the relevant columns to numpy arrays
    arrays = {col: df[col].to_numpy(copy=True) for col in group_cols}

    # Determine which rows are "in range" vs. "out of range"
    valid_mask = (
        (arrays[col_to_filter] >= lower_bound) &
        (arrays[col_to_filter] <= upper_bound)
    )

    # Keep only the valid (in-range) elements and shift them up
    for col in group_cols:
        valid_data = arrays[col][valid_mask]
        n_invalid = len(arrays[col]) - len(valid_data)

        # Shift up by overwriting the top part of the column with valid_data
        # and filling the remainder with None (or np.nan) at the bottom.
        new_col = np.concatenate([valid_data, [None]*n_invalid])

        # Assign back into df
        df[col] = new_col

    return df

def compute_gmm_responsibilities(data_col, GMM_guess):
    """
    Compute GMM responsibilities and updated mixing weights.

    Parameters:
    - data_col: pd.Series or np.ndarray (1D data)
    - means: list or array of component means (length K)
    - stds: list or array of component std deviations (length K)
    - weights: list or array of mixing weights (length K)

    Returns:
    - gamma: responsibility matrix (N x K)
    - updated_weights: array of updated mixing weights (length K)
    """
    X = data_col.values if isinstance(data_col, pd.Series) else data_col
    
    try:
        X = np.asarray(X, dtype=np.float64) 
    except ValueError:
        print(X)
        raise TypeError("Input data could not be converted to a numerical type (float64). Check for non-numeric entries.")
    
    if np.isnan(X).any() or np.isinf(X).any():
        raise ValueError("X contains NaN or Inf values.")
    means, stds, weights = GMM_guess
    means = np.array(means)
    stds = np.array(stds)
    weights = np.array(weights)
    
    N = len(X)
    K = len(means)
    
    # Step 1: Compute numerator of gamma_ik: π_k * N(x_i | μ_k, σ_k^2)
    gamma_numerators = np.zeros((N, K))
    for k in range(K):
        gamma_numerators[:, k] = weights[k] * stats.norm.pdf(X, loc=means[k], scale=stds[k])
    
    # gamma_numerators[gamma_numerators == 0] = 1e-10
    # Step 2: Normalize across components (sum over k)
    gamma_denominator = np.sum(gamma_numerators, axis=1, keepdims=True)
    # gamma_denominator[gamma_denominator == 0] = 1e-10
    gamma_denominator_safe = np.where(gamma_denominator == 0, 1e-10, gamma_denominator)
    gamma = gamma_numerators / gamma_denominator_safe
    if np.isnan(gamma).any():
        print("gamma_numerators:\n", np.argwhere(np.isnan(gamma_numerators)))

        # raise ValueError("Gamma contains NaN. Check your inputs and numerical stability.")

    # Step 3: Update mixing weights: π_k = (1/N) * sum_i gamma_ik
    updated_weights = np.mean(gamma, axis=0)
    # print(np.mean(gamma[:][1]))
    
    return gamma, updated_weights

def calculate_component_statistics(df, at_col, tt_col, lda_col, GMMx_weights=None):
    """
    Calculates statistical properties for a single velocity component.

    Parameters
    ----------
    df : pd.DataFrame => The DataFrame containing the data.
    at_col : str => Column name for "AT [ms]" (Arrival Time).
    tt_col : str => Column name for "TT [µs]" (Transit Time).
    lda_col : str => Column name for "LDA1 [m/s]" (Velocity).

    Returns
    -------
    tuple: A tuple containing:
        - counts (int)
        - at_min (float)
        - at_max (float)
        - d_rate (float)
        - tt_sum (float)
        - avg_vel (float)
        - rms_vel (float)
        - mean_conf (float)
        - rms_conf (float)
    """
    counts = df[tt_col].dropna().count()
    at_max = df[at_col].dropna().max()
    at_min = df[at_col].dropna()[df[at_col].dropna() > 0].min()

    d_rate = np.nan
    if at_max is not np.nan and at_min is not np.nan and (at_max - at_min) > 0:
        d_rate = counts * 1000 / (at_max - at_min)

    tt_sum = df[tt_col].sum()
    t_lda_sum = (df[tt_col] * df[lda_col]).sum()
    if GMMx_weights:
        vel, stds, _ = GMMx_weights
        vel = np.array(vel)
        stds = np.array(stds)
        gamma, w = compute_gmm_responsibilities(df['LDA1 [m/s]'].dropna(), GMMx_weights)
        # print(w)
        avg_vel = np.sum(w * vel)
    else:
        avg_vel = t_lda_sum / tt_sum if tt_sum != 0 else np.nan

    rms_vel = np.nan
    mean_conf = np.nan
    rms_conf = np.nan

    if counts > 1 and tt_sum != 0:
        df['t(lda-avg)2'] = df[tt_col] * (df[lda_col] - avg_vel)**2
        if GMMx_weights:
            rms_vel = np.sqrt(np.sum(w * (stds**2 + (vel - avg_vel)**2)))
        else:
            rms_vel = np.sqrt(df['t(lda-avg)2'].sum() / tt_sum)

        t_lda2_sum = (df[tt_col] * (df[lda_col]**2)).sum()
        lda2_dash = ((t_lda2_sum - 2 * avg_vel * t_lda_sum + (avg_vel**2) * tt_sum) * counts) / (tt_sum * (counts - 1))
        
        df_deg = counts - 1
        t_dist = stats.t.ppf(0.975, df_deg)
        mean_conf = t_dist * np.sqrt(lda2_dash / counts)
        rms_conf = t_dist * np.sqrt(lda2_dash / (2 * counts))

    return counts, at_min, at_max, d_rate, tt_sum, avg_vel, rms_vel, mean_conf, rms_conf

#%% Main function
def analyze_lda_data(directory_path, main_file_name, file_suffixes, num_components,
                     vx_range=None, vy_range=None, GMMx_weights=None, GMMy_weights=None):
    """
    Analyzes LDA (Laser Doppler Anemometry) data from specified Excel files,
    calculates various statistics, prints them, and generates histograms.

    Parameters
    ----------
    directory_path : str => The path to the directory containing the data files.
    main_file_name : str => The base name of the data files (e.g., "20250508_US_50mm_inc_Full").
    file_suffixes : list of str => A list of suffixes (e.g., "001", "002") to append to the 
                                   main_file_name to construct full file names.
    num_components : int => The number of velocity components to analyze (1 or 2).
    vx_range : list, optional => [lower_bound, upper_bound] for filtering LDA1 (U-component).
                                 If None, no filtering is applied.
    vy_range : list, optional => [lower_bound, upper_bound] for filtering LDA2 (V-component).
                                 If None, no filtering is applied.

    Returns
    -------
    list
        An array containing the calculated statistics.
        If num_components = 1: [counts1, "", D_rate, "", "", "", U_avg, "", U_rms, "",
                                mean_conf_u, "", rms_conf_u, ""]
        If num_components = 2: [counts1, counts2, D_rate, D_rate2, "", "", U_avg, V_avg, U_rms, 
                                V_rms, mean_conf_u, mean_conf_v, rms_conf_u, rms_conf_v]
    """

    Dfs = []
    total_at_duration_1 = 0
    total_at_duration_2 = 0
    
    fig, ax=plt.subplots(2, 1, figsize=(20, 15)) if num_components==2 else plt.subplots(1, 1, figsize=(20, 7))
    if num_components == 1: ax = [ax] # Make ax iterable for consistent indexing

    fig_title = "Histogram for points: " + histogram_title(file_suffixes)
    fig.suptitle(fig_title, fontsize=50)
    
    for n, suffix in enumerate(file_suffixes):
        original_file_path = os.path.join(directory_path, f"{main_file_name}{suffix}.xlsx")
        copy_file_path = os.path.join(directory_path, f"{main_file_name}{suffix}_copy.xlsx")

        print(f'Statistics for file: {os.path.basename(original_file_path)}')
        while True:
            try:
                Df = pd.read_excel(copy_file_path, skiprows=5)
            except Exception as e:
                print("Failed to open workbook; error: ")
                print(e)
                wingsbook = xw.Book(original_file_path)
                wingsapp = xw.apps.active
                wingsbook.save(copy_file_path)
                wingsapp.quit()
            else:
                break
            
        Df = pd.read_excel(copy_file_path, skiprows=5)
        
        # Apply filtering if ranges are provided
        if vx_range:
            LDA1 = ["AT [ms]", "TT [µs]", "LDA1 [m/s]"]
            Df = filter_and_shift(Df, LDA1, col_to_filter="LDA1 [m/s]",
                                  lower_bound=vx_range[0], upper_bound=vx_range[1])
            
        if num_components == 2 and vy_range:
            LDA2 = ["AT{2} [ms]", "TT{2} [µs]", "LDA2{2} [m/s]"]
            Df = filter_and_shift(Df, LDA2, col_to_filter="LDA2{2} [m/s]",
                                  lower_bound=vy_range[0], upper_bound=vy_range[1])
            
        # --- Calculate statistics for component 1 (U-component) ---
        (counts1, at_min_1, at_max_1, d_rate_1, tt_sum_1, u_avg, u_rms, mean_conf_u, rms_conf_u) = \
        calculate_component_statistics(Df, "AT [ms]", "TT [µs]", "LDA1 [m/s]")

        # Accumulate total active time duration
        if at_max_1 is not np.nan and at_min_1 is not np.nan and (at_max_1 - at_min_1) > 0:
            total_at_duration_1 += (at_max_1 - at_min_1) / 1000
        
        max_com1 = Df['LDA1 [m/s]'].dropna().max()
        min_com1 = Df['LDA1 [m/s]'].dropna().min()
        if not np.isnan(max_com1) and not np.isnan(min_com1):
            ax[0].set_xlim([min_com1-0.05*min_com1,  max_com1+0.05*max_com1])
        ax[0].hist(Df['LDA1 [m/s]'].dropna(), density=True, bins=25, edgecolor='black', alpha=0.5,
                   label=f'Run {n}, counts {counts1}')
        ax[0].set_xlabel('$u$ [m/s]')
        ax[0].set_ylabel('Probability Density')
        ax[0].legend()
        
        # Initialize variables for component 2 (V-component)
        counts2 = ""
        d_rate_2 = ""
        v_avg = ""
        v_rms = ""
        mean_conf_v = ""
        rms_conf_v = ""
        
        if num_components == 2:
            # --- Calculate statistics for component 2 (V-component) ---
            (counts2, at_min_2, at_max_2, d_rate_2, tt2_sum, v_avg, v_rms, mean_conf_v, rms_conf_v) = \
                calculate_component_statistics(Df, "AT{2} [ms]", "TT{2} [µs]", "LDA2{2} [m/s]")
            
            # Accumulate total active time duration for component 2
            if at_max_2 is not np.nan and at_min_2 is not np.nan and (at_max_2 - at_min_2) > 0:
                total_at_duration_2 += (at_max_2 - at_min_2) / 1000

            # Plot histograms for component 2
            max_com2 = Df['LDA2{2} [m/s]'].dropna().max()
            min_com2 = Df['LDA2{2} [m/s]'].dropna().min()
            if not np.isnan(max_com2) and not np.isnan(min_com2):
                ax[1].set_xlim([min_com2-0.05*min_com2,  max_com2+0.05*max_com2])
            ax[1].hist(Df['LDA2{2} [m/s]'].dropna(), density=True, bins=25, edgecolor='black', alpha=0.5,
                       label=f'Run {n}, counts {counts2}')
            ax[1].set_xlabel('$v$ [m/s]')
            ax[1].set_ylabel('Probability Density')
            ax[1].legend()
            
        # Print statistics for the current file
        row_format = "{:>15}" * (3 if num_components == 2 else 2)
        LDA_headers = ['LDA1', 'LDA2'] if num_components == 2 else ['LDA1']
        
        print(row_format.format("", *LDA_headers))
        print(row_format.format("Counts", counts1, counts2 if num_components == 2 else ""))
        print(row_format.format("AT-max", f'{at_max_1:0.2f}', f'{at_max_2:0.2f}' if num_components == 2 else ""))
        print(row_format.format("Data-rate", f'{d_rate_1:0.2f}', f'{d_rate_2:0.2f}' if num_components == 2 else ""))
        print(row_format.format("Ū" if num_components == 1 else "Ū", f'{u_avg:0.4f}', f'{v_avg:0.4f}' if num_components == 2 else ""))
        print(row_format.format("σ", f'{u_rms:0.4f}', f'{v_rms:0.4f}' if num_components == 2 else ""))
        print(row_format.format("ε(Ū)", f'{mean_conf_u:0.4f}', f'{mean_conf_v:0.4f}' if num_components == 2 else ""))
        print(row_format.format("ε(σᵤ)", f'{rms_conf_u:0.4f}', f'{rms_conf_v:0.4f}' if num_components == 2 else ""))
        print("-" * 50) # Separator for individual file stats

        Dfs.append(Df)
    
    ## Overall Data Statistics
    print(f"\nFull data statistics for points {histogram_title(file_suffixes)}:")
    concatenated_df = pd.concat(Dfs)

    # Recalculate global counts and data rates
    counts1_global = concatenated_df['TT [µs]'].dropna().count()
    counts2_global = concatenated_df['TT{2} [µs]'].dropna().count() if num_components == 2 else ""

    d_rate_global_1 = counts1_global / total_at_duration_1 if total_at_duration_1 > 0 else np.nan
    d_rate_global_2 = counts2_global / total_at_duration_2 if num_components == 2 and total_at_duration_2 > 0 else np.nan

    # Calculate global statistics for component 1    
    (counts1_global, _, _, _, tt_sum_global_1, u_avg_global, u_rms_global, mean_conf_u_global, rms_conf_u_global) = \
        calculate_component_statistics(concatenated_df, "AT [ms]", "TT [µs]", "LDA1 [m/s]", GMMx_weights)
        
    # Initialize variables for global component 2 statistics
    v_avg_global = ""
    v_rms_global = ""
    mean_conf_v_global = ""
    rms_conf_v_global = ""

    if num_components == 2:
        # Calculate global statistics for component 2
        (counts2_global, _, _, _, tt2_sum_global, v_avg_global, v_rms_global, mean_conf_v_global, rms_conf_v_global) = \
        calculate_component_statistics(concatenated_df, "AT{2} [ms]", "TT{2} [µs]", "LDA2{2} [m/s]", GMMy_weights)
    
    max_com1 = concatenated_df['LDA1 [m/s]'].dropna().max()
    min_com1 = concatenated_df['LDA1 [m/s]'].dropna().min()
    # Plot KDE for overall data
    
    concatenated_df['LDA1 [m/s]'].plot.kde(ax=ax[0], lw=5, color='k', label='_nolegend_')
    ax[0].axvline(u_avg_global, ls='-.', color='r', lw=3)
    ax[0].axvline(u_avg_global + u_rms_global, ls='--', color='r', lw=3)
    ax[0].axvline(u_avg_global - u_rms_global, ls='--', color='r', lw=3)
    if not np.isnan(max_com1) and not np.isnan(min_com1):
        ax[0].set_xlim([min_com1-0.05*min_com1,  max_com1+0.05*max_com1])
        
    if num_components == 2:
        max_com2 = concatenated_df['LDA2{2} [m/s]'].dropna().max()
        min_com2 = concatenated_df['LDA2{2} [m/s]'].dropna().min()
        # concatenated_df['LDA1 [m/s]'].plot.kde(ax=ax[0], lw=5, color='k', label='_nolegend_')
        concatenated_df['LDA2{2} [m/s]'].plot.kde(ax=ax[1], lw=5, color='k', label='_nolegend_')
        if not np.isnan(max_com2) and not np.isnan(min_com2):
            ax[1].set_xlim([min_com2-0.05*min_com2,  max_com2+0.05*max_com2])
        # ax[0].axvline(u_avg_global, ls='-.', color='r', lw=3)
        # ax[0].axvline(u_avg_global + u_rms_global, ls='--', color='r', lw=3)
        # ax[0].axvline(u_avg_global - u_rms_global, ls='--', color='r', lw=3)
        ax[1].axvline(v_avg_global, ls='-.', color='r', lw=3)
        ax[1].axvline(v_avg_global + v_rms_global, ls='--', color='r', lw=3)
        ax[1].axvline(v_avg_global - v_rms_global, ls='--', color='r', lw=3)
        
    
    
    # Create a separate figure for the overall histogram and KDE
    fig_overall, ax_overall = plt.subplots(2, 1, figsize=(20, 15)) if num_components == 2 else plt.subplots(1, 1, figsize=(20, 7))
    if num_components == 1:  ax_overall = [ax_overall]

    ax_overall[0].hist(concatenated_df['LDA1 [m/s]'].dropna(), density=True, bins=25, edgecolor='black')
    concatenated_df['LDA1 [m/s]'].plot.kde(ax=ax_overall[0], lw=5, color='k')
    ax_overall[0].set_xlabel('$U$ [m/s]')
    ax_overall[0].set_ylabel('Probability Density')
    ax_overall[0].set_title(f'Overall Histogram for LDA1 - Counts: {counts1_global}')

    if num_components == 2:
        ax_overall[1].hist(concatenated_df['LDA2{2} [m/s]'].dropna(), density=True, bins=25, edgecolor='black')
        concatenated_df['LDA2{2} [m/s]'].plot.kde(ax=ax_overall[1], lw=5, color='k')
        ax_overall[1].set_xlabel('$V$ [m/s]')
        ax_overall[1].set_ylabel('Probability Density')
        ax_overall[1].set_title(f'Overall Histogram for LDA2 - Counts: {counts2_global}')

    # Print global statistics
    row_format = "{:>15}" * (3 if num_components == 2 else 2)
    LDA_headers = ['LDA1', 'LDA2'] if num_components == 2 else ['LDA1']
    
    print(row_format.format("", *LDA_headers))
    print(row_format.format("Counts", counts1_global, counts2_global if num_components == 2 else ""))
    print(row_format.format("Data rate", f'{d_rate_global_1:0.2f}', f'{d_rate_global_2:0.2f}' if num_components == 2 else ""))
    print(row_format.format("Ū" if num_components == 1 else "Ū", f'{u_avg_global:0.8f}', f'{v_avg_global:0.8f}' if num_components == 2 else ""))
    print(row_format.format("σ", f'{u_rms_global:0.8f}', f'{v_rms_global:0.8f}' if num_components == 2 else ""))
    
    t1_global_display = f'{stats.t.ppf(0.975, counts1_global - 1):0.4f}' if counts1_global > 1 else 'N/A'
    t2_global_display = f'{stats.t.ppf(0.975, counts2_global - 1):0.4f}' if num_components == 2 and counts2_global > 1 else 'N/A'
    print(row_format.format("t-distribution", t1_global_display, t2_global_display))
    
    print(row_format.format("ε(Ū)", f'{mean_conf_u_global:0.8f}', f'{mean_conf_v_global:0.8f}' if num_components == 2 else ""))
    print(row_format.format("ε(σᵤ)", f'{rms_conf_u_global:0.8f}', f'{rms_conf_v_global:0.8f}' if num_components == 2 else ""))

    # Prepare the return array
    if num_components == 1:
        result_array = [
            counts1_global, "", d_rate_global_1, "", "", "",
            u_avg_global, "", u_rms_global, "",
            mean_conf_u_global, "", rms_conf_u_global, ""
        ]
    elif num_components == 2:
        result_array = [
            counts1_global, counts2_global, d_rate_global_1, d_rate_global_2, "", "",
            u_avg_global, v_avg_global, u_rms_global, v_rms_global,
            mean_conf_u_global, mean_conf_v_global, rms_conf_u_global, rms_conf_v_global
        ]
    else:
        raise ValueError("num_components must be 1 or 2.")

    return result_array
