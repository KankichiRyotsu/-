#!/usr/bin/python
"""
Vibrational Density of States (VDOS) Analysis

This script processes LAMMPS MD trajectories with velocity information to calculate VDOS. 

The expected dump file format is: ID TYPE x y z vx vy vz. 

It utilizes numpy.correlate and numpy.fft for efficiency. Caution is advised for large datasets 
due to memory usage. For mass-weighted VACF, a mapping file of atomic IDs to masses is required.

"""

__author__ = "Ryotsu Kankichi"
__creation_date__ = "Aug 3 2025"
__units__ = {
    "Time": "seconds",
    "Frequency": "THz",
}

import argparse
import glob
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process LAMMPS dump files to calculate VDOS.",
        epilog="NOTES: Program outputs raw files VACF.dat and VDOS.dat as well as figures VDOS.png and VACF.png",
        usage="Example, dump2VDOS.py --dump_file argon.dump --timestep 10.0e-15 --correlation_length 150",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dump_file", type=str, help="Path to the LAMMPS dump file.")
    group.add_argument(
        "--dump_pattern", type=str, help="Pattern for multiple LAMMPS dump files."
    )

    parser.add_argument(
        "--num_columns",
        type=int,
        default=8,
        help="Number of columns in ITEM: ATOMS. Note that vx, vy, and vz must be columns 6-8 (5-7 py)",
    )

    parser.add_argument(
        "--n_time_origins", "-n0",
        type=int,
        default=100,
        metavar="N",
        help=(
            "Number of time origins to sample per file "
            "(default: 100, evenly spaced in the usable range)."
        ),
    )

    parser.add_argument(
        "--sim_time_step",
        type=float,
        default=1.0e-15,
        help="Timestep in seconds (e.g., 1 fs = 1e-15 s ) uing MD calcculation",
    )

    parser.add_argument(
        "--mass_map_file",
        type=str,
        default=None,
        help="Path to the file mapping atom IDs to masses. Useful for mass weighted VDOS.",
    )

    parser.add_argument(
        "--correlation_length",
        type=int,
        default=500,
        help="Correlation length. This is the number of snapshots to use.",
    )

    # New arguments for windowing function and factors
    parser.add_argument(
        "--window_function",
        type=str,
        default="gaussian",
        choices=["gaussian", "hanning", "hamming", "blackman", "bartlett"],
        help="Windowing function to use for VDOS calculation.",
    )

    parser.add_argument(
        "--std_factor",
        type=float,
        default=15.5,
        help="Standard deviation factor for Gaussian windowing.",
    )

    parser.add_argument(
        "--pad_factor",
        type=int,
        default=15,
        help="Padding factor for FFT calculation in VDOS.",
    )

    parser.add_argument(
        "--vacf_image_file",
        type=str,
        default="VACF.png",
        help="VACF result visualized",
    )

    parser.add_argument(
        "--vdos_image_file",
        type=str,
        default="VDOS.png",
        help="VDOS result visualized",
    )

    parser.add_argument(
        "--vacf_data_file",
        type=str,
        default="VACF.dat",
        help="VACF result data",
    )

    parser.add_argument(
        "--vdos_data_file",
        type=str,
        default="VDOS.dat",
        help="VDOS result data",
    )

    args = parser.parse_args()

    # Check for conditional requirements
    if args.dump_pattern:
        if args.num_atoms is None or args.num_snapshots is None:
            parser.error(
                "--num_atoms and --num_snapshots are required when --dump_pattern is provided"
            )

    return args


def read_lammps_dump(filename, num_atoms, num_fields, num_snapshots):
    """
    Read data from a LAMMPS dump file.

    Parameters:
    - filename (str): Path to the LAMMPS dump file.
    - num_atoms (int): Number of atoms to read from each snapshot.
    - num_fields (int): Number of data fields for each atom in the dump file.
    - num_snapshots (int): Number of snapshots present in the dump file.

    Returns:
    - data (ndarray): A numpy array containing the data from the dump file.
                      The array has a shape of (num_atoms, num_fields, num_snapshots).
    """
    data = np.ndarray((num_atoms, num_fields, num_snapshots), dtype=float)

    with open(filename, "r") as file:
        for snapshot_index in range(num_snapshots):
            # Skip header lines
            for _ in range(9):  # Assuming there are 8 lines in the header
                file.readline()

            # Read data for each atom in the snapshot
            for atom_index in range(num_atoms):
                line = file.readline().strip().split()
                data[atom_index, :, snapshot_index] = [float(field) for field in line]

    return data


def infer_from_dump(file_name):
    """
    Infer number of atoms, timestep difference, and total snapshots
    from a LAMMPS text dump (ASCII) file.

    Returns
    -------
    num_atoms       : int
    timestep_diff   : int  (0 if only one snapshot)
    total_snapshots : int
    """
    num_atoms = None
    first_ts = None
    second_ts = None
    last_ts = None
    snapshot_count = 0

    with open(file_name) as fh:
        for line in fh:
            # --- TIMESTEP ---------------------------------------------------
            if line.startswith("ITEM: TIMESTEP"):
                ts = int(next(fh).strip())
                snapshot_count += 1

                if first_ts is None:
                    first_ts = ts
                elif second_ts is None:
                    second_ts = ts
                last_ts = ts
                continue

            # --- NUMBER OF ATOMS -------------
            if num_atoms is None and line.startswith("ITEM: NUMBER OF ATOMS"):
                num_atoms = int(next(fh).strip())

    if snapshot_count <= 1:
        timestep_diff = 0
        total_snaps   = snapshot_count
    else:
        timestep_diff = second_ts - first_ts 
        total_snaps   = snapshot_count

    return num_atoms, total_snaps, timestep_diff,


def parsemapfile(mass_map_file:str) -> dict:
    """
    Reads LAMMPS atom-type mass file and returns a dictionary
    mapping atom-type IDs to their masses.

    Lines starting with "#" are treated as comments and ignored.

    example of mass map file
    ---
    # ID Mass
    1 12.011
    2 1.008
    3 15.999    
    ---
    """

    # Load data ignoring comment lines starting with '#'
    data = np.loadtxt(mass_map_file, comments="#")

    # Create a dictionary mapping atom IDs (as integers) to masses
    massmap = {int(atom_id): mass for atom_id, mass in data}

    return massmap


def autocorr(X):
    """the convolution is actually being done here
    meaning from -inf to inf so we only want half the
    array"""

    result = np.correlate(X, X, mode="full")
    return result[result.size // 2 :]


def fft_autocorr(AutoCorr, dt):
    """FFT of autocorrelation function"""
    # fft_arry = fftpack.dct(AutoCorr)*dt
    fft_arry = np.fft.rfft(AutoCorr) * dt
    return fft_arry


def calculate_vacf_ensemble_average_2(
    file_pattern: str,
    max_lag: int,
    n_time_origins: int,
    num_atoms: int,
    num_fields: int,
    total_snapshots: int,
    mass_map: dict[int, float] | None = None,
):
    """
    Compute the ensemble-averaged velocity–autocorrelation function (VACF)
    and return it normalized so that VACF(0) == 1.

    Parameters
    ----------
    file_pattern : str
        Shell-style wildcard pattern for LAMMPS dump files
        (e.g. "dump.*" or "traj_*.lammpstrj").
    max_lag : int
        Maximum lag (in MD steps) for which the autocorrelation is evaluated.
        VACF will be returned on the range τ = 0 .. max_lag-1.
    n_time_origins : int
        Exact number of time-origins (t₀) to sample in *each* file.
        They are chosen evenly over the usable interval
        0 ≤ t₀ ≤ total_snapshots − max_lag.
    num_atoms : int
        Number of atoms in every snapshot (rows per snapshot in the dump).
    num_fields : int
        Number of numeric columns per atom in the dump
        (id type x y z vx vy vz  →  typically 8).
    total_snapshots : int
        Total number of snapshots contained in every dump file.
    mass_map : dict[int, float] | None, optional
        Mapping from atom type (or ID) to atomic mass.  If supplied, the VACF
        is mass-weighted by a single power of mᵢ.  If *None* (default) all
        masses are treated as 1.0.

    Returns
    -------
    times : ndarray, shape (max_lag,)
        Integer lag values 0, 1, 2, …, max_lag-1 (units: MD steps).
    vacf  : ndarray, shape (max_lag,)
        Normalized ensemble-averaged VACF such that vacf[0] == 1.
        Definition used here:
            vacf(τ) = 1/C(0) · (1/N_t0) Σ_{t0} Σ_i m_i
                      ⟨ v_i(t0) · v_i(t0+τ) ⟩
        where • is a dot product over x,y,z; mᵢ = 1 if mass_map is None.
    """
    if max_lag > total_snapshots:
        raise ValueError("`max_lag` must not exceed `total_snapshots`.")
    if n_time_origins <= 0:
        raise ValueError("`n_time_origins` must be positive.")

    # Column indices of vx, vy, vz in the dump (id type x y z vx vy vz)
    comp2idx = {"vx": 5, "vy": 6, "vz": 7}
    use_mass = mass_map is not None

    # Collect all files that match the pattern
    files = sorted(glob.glob(file_pattern))
    if not files:
        raise RuntimeError(f"No files match pattern '{file_pattern}'")

    # Output arrays
    times = np.arange(max_lag)                 # 0, 1, …, max_lag-1
    vacf_total = np.zeros(max_lag, float)      # accumulated over files

    max_start = total_snapshots - max_lag

    # Loop over each dump file ------------------------------------------------
    for fname in files:
        data = read_lammps_dump(fname, num_atoms, num_fields, total_snapshots)
        print(f"Processing file: {fname}")

        # Choose exactly n_time_origins evenly spaced start indices
        origins = np.unique(
            np.floor(np.linspace(0, max_start, n_time_origins)).astype(int)
        )

        vacf_file = np.zeros(max_lag, float)   # accumulator for this file

        # --- iterate over all chosen time-origins ---------------------------
        for t0 in origins:
            # For every atom, build a length-max_lag velocity series
            for i in range(num_atoms):
                atom_type = int(data[i, 1, t0])          # int key for dict
                mass_i = mass_map.get(atom_type, 1.0) if use_mass else 1.0

                # Velocity components, window [t0, t0+max_lag)
                vx = data[i, comp2idx["vx"], t0 : t0 + max_lag]
                vy = data[i, comp2idx["vy"], t0 : t0 + max_lag]
                vz = data[i, comp2idx["vz"], t0 : t0 + max_lag]

                # Autocorrelation for each component (lags ≥ 0)
                ac_vx = autocorr(vx)
                ac_vy = autocorr(vy)
                ac_vz = autocorr(vz)

                # dot product contribution, mass-weighted once
                vacf_file += mass_i * (ac_vx + ac_vy + ac_vz)

        # Average over all time-origins in this file
        vacf_file /= len(origins)

        # Add to global accumulator
        vacf_total += vacf_file

    # Average over files
    vacf_total /= len(files)

    # Normalize so VACF(0) = 1
    if vacf_total[0] != 0.0:
        vacf_total /= vacf_total[0]

    return times, vacf_total


def write_vacf(time, vacf, filename="VACF.dat"):
    f = open(filename, "w")
    f.write("# Time(ps) Norm. Int.(No Units) \n")
    for i in range(time.size):
        f.write("%f %f \n" % (time[i], vacf[i]))
    f.close()
    return None


def plot_vacf(time, vacf, image_file="VACF.png"):
    """
    Plot the Velocity Auto-Correlation Function (VACF).

    Parameters:
    - time: ndarray, array of time values.
    - vacf: ndarray, array of VACF values.
    """
    plt.plot(time, vacf, "k")
    plt.xlim((0, None))
    plt.ylabel(r"$\frac{v(t)\cdot v(t)}{v(0)\cdot v(0)}$", fontsize=20)
    plt.xlabel("Time [ps]")
    plt.savefig(image_file)
    plt.close()


def write_vdos(freq, vdos, filename="VDOS.dat"):
    """
    Write the VDOS data (frequency and FFT of VACF) to a file.

    Parameters:
    - freq: ndarray, array of frequency values.
    - fft_v: ndarray, FFT of the VACF.
    - filename: str (optional), name of the file to write the data to.
    """
    with open(filename, "w") as file:
        file.write("# Frequency (THz), VDOS (THz^-1)\n")
        for f, fft_val in zip(freq, vdos):
            file.write(f"{f:.5f}, {fft_val.real:.5e}\n")


def get_vdos(
    vacf,
    corlen,
    dt,
    image_file="VDOS.png",
    data_file="VDOS.dat",
    window="gaussian",
    std_factor=15.5,
    pad_factor=15,
    freq_max=40.0,
):
    """
    Vibrational Density of States (VDOS) using FFT of the autocorrelation data.

     Parameters:
     - vacf: ndarray, Velocity Auto-Correlation Function data.
     - corlen: int, Correlation length.
     - dt: float, Timestep duration in femtoseconds.
    """

    # build gaussian window on original VACF length
    std = corlen / std_factor  # controls width of gaussian
    window = signal.get_window(("gaussian", std), vacf.size)  # length corlen

    # apply window to VACF first
    windowed = vacf * window  # shape: (corlen,)

    # then zero-pad to increase frequency resolution
    padded = np.pad(windowed, (0, vacf.size * pad_factor), mode="constant")  # now longer

    # FFT on the padded, windowed VACF
    freq = np.fft.rfftfreq(padded.size, d=dt) / 1.0e12  # THz
    fft_vacf = np.fft.rfft(padded) * dt * 1.0e12        # direct FFT scaling

    write_vdos(freq, fft_vacf, filename=data_file)

    # Plotting
    plt.plot(freq, np.abs(fft_vacf))
    plt.xlim((0, freq_max))
    plt.ylim((0, None))
    xx, locs = plt.xticks()
    ll = ["%2.0f" % a for a in xx]
    plt.xticks(xx, ll)
    plt.xlabel("THz")
    plt.ylabel(r"VDOS [THz $^{-1}$]")
    plt.savefig(image_file)
    plt.close()


def main():
    args = parse_arguments()

    # Extract variables from args
    if args.dump_file is not None:
        fil = args.dump_file
    else:
        fil = args.dump_pattern

    ncols = args.num_columns
    dt_sim = args.sim_time_step
    mass_map_file = args.mass_map_file
    n_time_origine = args.n_time_origins
    corlen = args.correlation_length
    window_function = args.window_function
    std_factor = args.std_factor
    pad_factor = args.pad_factor
    vacf_image_file = args.vacf_image_file
    vdos_image_file = args.vdos_image_file
    vacf_data_file = args.vacf_data_file
    vdos_data_file = args.vdos_data_file

    if args.dump_file:
        num_atoms, num_snapshots, time_diff = infer_from_dump(fil)
    dt = dt_sim * time_diff
    
    if mass_map_file != None:
        mass_map_dir = parsemapfile(mass_map_file)
    else:
        mass_map_dir = None

    x, vacf = calculate_vacf_ensemble_average_2(
        fil, 
        corlen, 
        n_time_origine, 
        num_atoms, 
        ncols, 
        num_snapshots, 
        mass_map=mass_map_dir,
    )

    time_ps = x * dt * 1.0e12

    write_vacf(time_ps, vacf, filename=vacf_data_file)
    plot_vacf(time_ps, vacf, image_file=vacf_image_file)

    get_vdos(
        vacf,
        corlen,
        dt,
        window=window_function,
        std_factor=std_factor,
        pad_factor=pad_factor,
        image_file=vdos_image_file,
        data_file=vdos_data_file,
    )

    return None


if __name__ == "__main__":
    main()
