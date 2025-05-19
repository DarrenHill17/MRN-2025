import pandas as pd
import os
import glob
from io import StringIO
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
import re

import numpy as np
from scipy.optimize import curve_fit

def fit_E_Bernoulli(dist_vals, load_vals, L, b, h):
    I = (b * h**3) / 12  # Moment of inertia

    # Convert distances to meters
    dist_vals_m = dist_vals / 1000

    def model_with_offset(F, E, delta0):
        # Pure bending deflection (center of simply supported beam)
        return (F * L**3) / (48 * E * I) + delta0

    # Initial guess for [E, delta0]
    initial_guess = [1E6, 0]

    # Fit model
    popt, _ = curve_fit(
        model_with_offset,
        load_vals,
        dist_vals_m,
        p0=initial_guess,
        bounds=([1e5, -10], [1e10, 10])
    )

    E_fit, delta0 = popt

    # Predicted deflections in meters
    predicted = model_with_offset(load_vals, E_fit, delta0)

    # Compute R²
    ss_res = np.sum((dist_vals_m - predicted) ** 2)
    ss_tot = np.sum((dist_vals_m - np.mean(dist_vals_m)) ** 2)
    r_squared = 1 - ss_res / ss_tot

    return E_fit, delta0, r_squared


def parse_custom_csv(filepath):
    sep = '\t'  # Tab separator

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Initialize sensor resolution
    sensor_resolution = None

    # Extract sensor resolution before the header
    for line in lines:
        if "Force Sensor Resolution" in line:
            parts = line.strip().split(":")
            if len(parts) == 2:
                sensor_resolution = parts[1].strip()

    # Find the index of the data header
    header_index = None
    for i, line in enumerate(lines):
        if line.strip().startswith("Reading"):
            header_index = i
            break

    if header_index is None:
        raise ValueError("Header row not found.")

    # Read the data from that point on
    data_str = "".join(lines[header_index:])
    df = pd.read_csv(StringIO(data_str), sep=sep)

    return df, sensor_resolution


def smooth_data(load_data, distance_data, window_size=51, poly_order=3, passes=1, failure_load_threshold=25, min_failure_distance=1.0):
    # Ensure window size is odd and valid
    if window_size % 2 == 0:
        window_size += 1
    if window_size >= len(load_data):
        window_size = len(load_data) - 1 if len(load_data) % 2 == 0 else len(load_data)

    # Step 1: Fully smooth the curve first
    smoothed = load_data.copy()
    for _ in range(passes):
        smoothed = savgol_filter(smoothed, window_length=window_size, polyorder=poly_order)

    # Step 2: Detect failure point
    peak_index = np.argmax(load_data)
    post_peak_load = load_data[peak_index:]
    failure_candidates = np.where(post_peak_load < failure_load_threshold)[0]

    if failure_candidates.size > 0:
        failure_index = peak_index + failure_candidates[0]
        failure_distance = distance_data[failure_index]

        if -failure_distance > min_failure_distance:  # Assuming distances are negative
            cutoff_distance = failure_distance + 0.5

            # Correct mask for negative distances: keep smoothed data before cutoff_distance
            mask = distance_data >= cutoff_distance
            result = smoothed.copy()
            result[~mask] = load_data[~mask]  # Replace smoothed data after cutoff with raw data
            return result

    return smoothed
    

def plot_curves(directory_path):
    csv_files = glob.glob(os.path.join(directory_path, "*.csv"))

    config_dict = {}
    for file in csv_files:
        match = re.search(r'Config\s*(\d+)-(\d+)', file)
        if match:
            config_num, specimen_num = match.groups()
            config_dict.setdefault(config_num, {})[int(specimen_num)] = file

    height_data = pd.read_excel("../DoE Results.xlsx")[[
    'Combination ID', 'Sample ID', 'Measured Height [mm]'
    ]].dropna()

    # Initialize data collection dictionaries
    sigma_f_data = {}
    E_f_data = {}
    x_data = {}

    for config_number, specimens in config_dict.items():
        for specimen_num, file in specimens.items():
            df, sensor_resolution = parse_custom_csv(file)
            df = df.iloc[:, :-1]
            df = df.drop(df.index[-1])

            if sensor_resolution == "1 N":
                df['Load [N]'] = smooth_data(df['Load [N]'].to_numpy(), df['Distance [mm]'].to_numpy())

            max_force = df['Load [N]'].max()
            max_force_index = df['Load [N]'].idxmax()
            distance_at_max_force = df.loc[max_force_index, 'Distance [mm]']

            dist_vals = df['Distance [mm]'].to_numpy()
            load_vals = df['Load [N]'].to_numpy()

            L = 64E-3
            b = 10E-3                
            h = height_data[
                (height_data['Combination ID'] == int(config_number)) & 
                (height_data['Sample ID'] == int(specimen_num))
            ]['Measured Height [mm]'].values[0] / 1000
            sigma_f = max_force * 3 * L / 2 / b / h**2

            mask = (-dist_vals <= -0.1 * distance_at_max_force)
            E1, delta0, r_sq = fit_E_Bernoulli(-dist_vals[mask], load_vals[mask], L, b, h)

            # Store data
            if config_number in sigma_f_data and sigma_f_data[config_number].size > 0: 
                sigma_f_data[config_number] = np.append(sigma_f_data[config_number], sigma_f/1E6)
            else:
                sigma_f_data[config_number] = np.array([sigma_f/1E6])

            if config_number in E_f_data and E_f_data[config_number].size > 0: 
                E_f_data[config_number] = np.append(E_f_data[config_number], E1/1E6)
            else:
                E_f_data[config_number] = np.array([E1/1E6])

            if config_number in x_data and x_data[config_number].size > 0: 
                x_data[config_number] = np.append(x_data[config_number], -distance_at_max_force)
            else:
                x_data[config_number] = np.array(-distance_at_max_force)

            I = (b * h**3) / 12
            A = b * h
            s = 1 / (L**3 / (48 * E1 * I)) / 1000
            steps = int(np.floor((max_force) / s))
            grad_x_vals = np.linspace(0, steps, steps + 1)
            intercept = -s * delta0 * 1000
            grad_y_vals = s * grad_x_vals + intercept

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(-dist_vals, load_vals, label='Raw Data')
            ax.plot(grad_x_vals, grad_y_vals, label=f'Initial Slope')
            ax.scatter(-distance_at_max_force, max_force, color='red', label='Max Force Point')
            ax.set_title(f"Config {config_number}-{specimen_num}", fontsize=12)
            ax.set_xlabel("Distance [mm]")
            ax.set_ylabel("Load [N]")
            ax.legend(fontsize=9, loc=4)
            ax.grid(True)

            if sensor_resolution == "1 N":
                annotation = (
                    f"σₙ: {sigma_f/1E6:.2f} MPa\n"
                    f"x_σₙ: {-distance_at_max_force:.3f} mm\n"
                    f"E: {E1/1E6:.2f} MPa\n"
                    f"R²: {r_sq:.4f}\n"
                    f"Smoothed"
                )
            else:
                annotation = (
                    f"σₙ: {sigma_f/1E6:.2f} MPa\n"
                    f"x_σₙ: {-distance_at_max_force:.3f} mm\n"
                    f"E: {E1/1E6:.2f} MPa\n"
                    f"R²: {r_sq:.4f}"
                )
            ax.text(
                0.98, 0.5, annotation,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment='center',
                horizontalalignment='right',
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.6)
            )

            output_dir = os.path.join(directory_path, "Plots\\Bernoulli Beam Model")
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"Config {config_number}-{specimen_num}.png")
            plt.tight_layout()
            plt.savefig(output_file, dpi=300)
            plt.show()
            plt.close()

    # Return summary data if needed
    return sigma_f_data, E_f_data, x_data


def plot_error_bars(input_data, plot_title, x_axis_label = '[MPa]'):
    xtick_labels = []
    xtick_positions = []
    for row in enumerate(input_data.items()):
        config = row[1][0]
        data = row[1][1]
        mean = np.mean(data)
        std_dev = np.std(data, ddof=1)
        std_err = std_dev / np.sqrt(len(data))

        print(mean)
    
        plt.errorbar(row[0], mean, yerr=std_dev, fmt='o', capsize=10)
    
        xtick_labels.append(f'{config}')
        xtick_positions.append(row[0])
        
    plt.xticks(xtick_positions, xtick_labels)
    plt.xlabel("Config")
    plt.ylabel(x_axis_label)
    plt.title(plot_title)
    plt.grid(True)
    plt.show()
    

def is_pareto_efficient(points):
    is_efficient = np.ones(points.shape[0], dtype=bool)
    for i, c in enumerate(points[:, :2]):  # Only consider x and y for Pareto comparison
        if is_efficient[i]:
            is_efficient[is_efficient] = (
                np.any(points[is_efficient, :2] < c, axis=1) |
                np.all(points[is_efficient, :2] == c, axis=1)
            )
            is_efficient[i] = True
    return is_efficient

def draw_pareto_with_background(points, title, xlabel):
    # Compute Pareto front
    pareto_mask = is_pareto_efficient(points)
    pareto_points = points[pareto_mask]
    pareto_points = pareto_points[np.argsort(pareto_points[:, 0])]  # Sort by x-axis

    # Precompute axis limits
    x_range = points[:, 0].max() - points[:, 0].min()
    y_range = points[:, 1].max() - points[:, 1].min()
    x_pad = 0.05 * x_range
    y_pad = 0.05 * y_range
    x_min, x_max = points[:, 0].min() - x_pad, points[:, 0].max() + x_pad
    y_min, y_max = points[:, 1].min() - y_pad, points[:, 1].max() + y_pad

    # Build step-like outline of the dominated region
    x = pareto_points[:, 0]
    y = pareto_points[:, 1]
    x_step = np.concatenate(([x_min], x, [x_max]))
    y_step = np.concatenate(([y_max], y, [y_max]))

    # Plot
    plt.figure(figsize=(8, 6))
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    
    plt.fill_between(x_step, y_step, y_max, step='post', color='red', alpha=0.2, label='Dominated Region')
    plt.scatter(points[~pareto_mask][:, 0], points[~pareto_mask][:, 1], color='grey', alpha=0.4, label='Dominated Points')
    plt.plot(x, y, linestyle='--', color='black', linewidth=1.2, zorder=1, label='Pareto Front')
    plt.scatter(x, y, facecolors='yellow', edgecolors='black', linewidths=1.0, s=60, label='Pareto Points')

    # Annotate using Combination ID (stored in the 3rd column)
    for i, config in enumerate(pareto_points):
        x_p, y_p, combo_id = config
        plt.text(x_p+0.2, y_p+2, f'{int(combo_id)}', color='black', fontsize=9, ha='center', va='center')
    
    # Final plot settings
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Time (minimize)")
    plt.grid(True)
    plt.legend()
    plt.show()
