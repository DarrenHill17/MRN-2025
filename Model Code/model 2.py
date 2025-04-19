import pandas as pd
import os
import glob
from io import StringIO
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import re

def fit_E_G_with_offset(dist_vals, load_vals, L, b, h, kappa=5/6):
    A = b * h

    # Convert distance to meters
    dist_vals_m = dist_vals / 1000

    def model_with_offset(F, E, nu, delta0):
        inv_stiffness = (L**3) / (4 * b * h**3) + (L * (12 + 11*nu)) / (20 * b * h)
        return F/E * inv_stiffness + delta0

    initial_guess = [1E6, 0.3, 0]

    # Fit model
    popt, _ = curve_fit(
        model_with_offset,
        load_vals,
        dist_vals_m,
        p0=initial_guess,
        bounds=([1e5, 0, -np.inf], [1e10, 1, np.inf])
    )

    E_fit, nu_fit, delta0 = popt

    predicted = model_with_offset(load_vals, E_fit, nu_fit, delta0)

    ss_res = np.sum((dist_vals_m - predicted) ** 2)
    ss_tot = np.sum((dist_vals_m - np.mean(dist_vals_m)) ** 2)
    r_squared = 1 - ss_res / ss_tot

    return E_fit, nu_fit, delta0, r_squared


def parse_custom_csv(filepath):
    sep = '\t'

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    sensor_resolution = None

    for line in lines:
        if "Force Sensor Resolution" in line:
            parts = line.strip().split(":")
            if len(parts) == 2:
                sensor_resolution = parts[1].strip()

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


def plot_curves(directory_path):
    csv_files = glob.glob(os.path.join(directory_path, "*.csv"))

    config_dict = {}
    for file in csv_files:
        match = re.search(r'Config\s*(\d+)-(\d+)', file)
        if match:
            config_num, specimen_num = match.groups()
            config_dict.setdefault(config_num, {})[int(specimen_num)] = file

    sigma_f_data = {}
    sigma_f_avgd_data = {}
    nu_data = {}
    E_f_data = {}
    G_data = {}

    for config_number, specimens in config_dict.items():
        for specimen_num, file in specimens.items():
            df, sensor_resolution = parse_custom_csv(file)
            df = df.iloc[:, :-1]
            df = df.drop(df.index[-1])

            max_force = df['Load [N]'].max()
            max_force_index = df['Load [N]'].idxmax()
            distance_at_max_force = df.loc[max_force_index, 'Distance [mm]']
            mask = (df['Distance [mm]'] >= 1.05 * distance_at_max_force) & (df['Distance [mm]'] <= 0.95 * distance_at_max_force)
            max_force_averaged = df.loc[mask, 'Load [N]'].mean()

            dist_vals = df['Distance [mm]'].to_numpy()
            load_vals = df['Load [N]'].to_numpy()

            L = 64E-3
            b, h = 10E-3, 4E-3
            sigma_f = max_force * 3 * L / 2 / b / h**2
            sigma_f_avg = max_force_averaged * 3 * L / 2 / b / h**2

            mask = (-dist_vals <= -0.1 * distance_at_max_force)
            E1, nu1, delta0, r_sq = fit_E_G_with_offset(-dist_vals[mask], load_vals[mask], L, b, h)
            G = E1 / 2 / (1 + nu1)

            # Store data
            if config_number in sigma_f_data and sigma_f_data[config_number].size > 0: 
                sigma_f_data[config_number] = np.append(sigma_f_data[config_number], sigma_f/1E6)
            else:
                sigma_f_data[config_number] = np.array([sigma_f/1E6])

            if config_number in sigma_f_avgd_data and sigma_f_avgd_data[config_number].size > 0: 
                sigma_f_avgd_data[config_number] = np.append(sigma_f_avgd_data[config_number], sigma_f_avg/1E6)
            else:
                sigma_f_avgd_data[config_number] = np.array([sigma_f_avg/1E6])

            if config_number in nu_data and nu_data[config_number].size > 0: 
                nu_data[config_number] = np.append(nu_data[config_number], nu1)
            else:
                nu_data[config_number] = np.array([nu1])

            if config_number in E_f_data and E_f_data[config_number].size > 0: 
                E_f_data[config_number] = np.append(E_f_data[config_number], E1/1E6)
            else:
                E_f_data[config_number] = np.array([E1/1E6])

            if config_number in G_data and G_data[config_number].size > 0: 
                G_data[config_number] = np.append(G_data[config_number], G/1E6)
            else:
                G_data[config_number] = np.array([G/1E6])

            I = (b * h**3) / 12
            A = b * h
            kappa = 10 * (1 + nu1) / (12 + 11 * nu1)
            s = 1 / ((L**3 / (48 * E1 * I)) + (L / (4 * kappa * A * G))) / 1000
            steps = int(np.floor((max_force) / s))
            grad_x_vals = np.linspace(0, steps, steps + 1)
            intercept = -s * delta0 * 1000
            grad_y_vals = s * grad_x_vals + intercept

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(-dist_vals, load_vals, label='Raw Data')
            ax.plot(grad_x_vals, grad_y_vals, label=f'Initial Slope (R²={r_sq:.4f})')
            ax.scatter(-distance_at_max_force, max_force, color='red', label='Max Force Point')
            ax.set_title(f"Config {config_number}-{specimen_num}", fontsize=12)
            ax.set_xlabel("Distance [mm]")
            ax.set_ylabel("Load [N]")
            ax.legend(fontsize=9, loc=4)
            ax.grid(True)

            annotation = (
                f"σₙ: {sigma_f/1E6:.2f} MPa\n"
                f"σₙ_avg: {sigma_f_avg/1E6:.2f} MPa\n"
                f"E: {E1/1E6:.2f} MPa\n"
                f"G: {G/1E6:.2f} MPa\n"
                f"ν: {nu1:.3f}\n"
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

            output_dir = os.path.join(directory_path, "Plots")
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"Config {config_number}-{specimen_num}.png")
            plt.tight_layout()
            plt.savefig(output_file, dpi=300)
            plt.show()
            plt.close()

    return sigma_f_data, sigma_f_avgd_data, nu_data, E_f_data, G_data


def plot_error_bars(input_data, plot_title):
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
    plt.ylabel("[MPa]")
    plt.title(plot_title)
    plt.grid(True)
    plt.show()
