"""
Compute per-atom VDOS/PDOS from a LAMMPS dump file in GPUMD-like dos.out format.

v2 optimizations:
- optional `assume_sorted_ids` fast path
- optional `np.fromstring` ATOMS block parser

References:
- https://gpumd.org/dev/gpumd/input_parameters/compute_dos.html
- https://gpumd.org/gpumd/output_files/dos_out.html
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Literal, Mapping, Optional, Tuple

import numpy as np


def _trapz(y: np.ndarray, x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compatibility wrapper for trapezoidal integration across NumPy versions."""
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x, axis=axis)  # type: ignore[attr-defined]
    return np.trapz(y, x, axis=axis)


class LammpsDumpReader:
    """
    Read a LAMMPS dump (lammpstrj) and build a velocity tensor.

    The reader enforces atom order to ascending `id` and keeps this order
    consistent across all retained frames.
    """

    REQUIRED_ATOM_COLUMNS = ("id", "type", "x", "y", "z", "vx", "vy", "vz")

    def __init__(
        self,
        dump_path: str | Path,
        dtype: np.dtype | type = np.float32,
        use_memmap: bool = False,
        memmap_dir: Optional[str | Path] = None,
        assume_sorted_ids: bool = False,
        use_fromstring_parser: bool = True,
    ) -> None:
        """
        Initialize a dump reader.

        Parameters
        ----------
        dump_path
            Path to the LAMMPS dump file.
        dtype
            Floating dtype for the stored velocity tensor.
        use_memmap
            If True, allocate velocity tensor as np.memmap on disk.
        memmap_dir
            Directory where temporary memmap file is placed.
        assume_sorted_ids
            If True, assume every retained frame is already sorted by atom id
            and skip per-frame id remapping and duplicate checks.
        use_fromstring_parser
            If True, parse ATOMS blocks using `np.fromstring` for speed.
        """
        self.dump_path = Path(dump_path)
        self.dtype = np.dtype(dtype)
        self.use_memmap = use_memmap
        self.memmap_dir = Path(memmap_dir) if memmap_dir is not None else None
        self.assume_sorted_ids = bool(assume_sorted_ids)
        self.use_fromstring_parser = bool(use_fromstring_parser)
        self._memmap_path: Optional[Path] = None

    def cleanup(self) -> None:
        """Remove temporary memmap backing file created by this reader."""
        if self._memmap_path is not None and self._memmap_path.exists():
            self._memmap_path.unlink()
        self._memmap_path = None

    def _parse_atoms_header(self, header_line: str, frame_idx: int) -> Dict[str, int]:
        """Parse `ITEM: ATOMS ...` header and return a column-name to index map."""
        tokens = header_line.strip().split()
        if len(tokens) < 3 or tokens[0] != "ITEM:" or tokens[1] != "ATOMS":
            raise ValueError(
                f"Frame {frame_idx}: malformed ATOMS header: {header_line.strip()!r}"
            )
        columns = tokens[2:]
        col_map = {name: i for i, name in enumerate(columns)}
        missing = [c for c in self.REQUIRED_ATOM_COLUMNS if c not in col_map]
        if missing:
            raise ValueError(
                f"Frame {frame_idx}: ATOMS columns missing required fields: {missing}. "
                f"Found columns: {columns}"
            )
        return col_map

    def _read_frame_preamble(
        self, fh, frame_idx: int
    ) -> Optional[Tuple[int, Dict[str, int]]]:
        """
        Read a frame header and return (n_atoms, col_map), or None at EOF.
        """
        line = fh.readline()
        if not line:
            return None
        if not line.startswith("ITEM: TIMESTEP"):
            raise ValueError(
                f"Frame {frame_idx}: expected 'ITEM: TIMESTEP', got: {line.strip()!r}"
            )
        timestep_line = fh.readline()
        if not timestep_line:
            raise ValueError(f"Frame {frame_idx}: unexpected EOF after TIMESTEP header")

        number_header = fh.readline()
        if not number_header:
            raise ValueError(f"Frame {frame_idx}: unexpected EOF before NUMBER OF ATOMS")
        if number_header.strip() != "ITEM: NUMBER OF ATOMS":
            raise ValueError(
                f"Frame {frame_idx}: expected 'ITEM: NUMBER OF ATOMS', got: "
                f"{number_header.strip()!r}"
            )

        n_atoms_line = fh.readline()
        if not n_atoms_line:
            raise ValueError(
                f"Frame {frame_idx}: unexpected EOF while reading atom count"
            )
        try:
            n_atoms = int(n_atoms_line.strip())
        except ValueError as exc:
            raise ValueError(
                f"Frame {frame_idx}: invalid NUMBER OF ATOMS value: "
                f"{n_atoms_line.strip()!r}"
            ) from exc

        box_header = fh.readline()
        if not box_header:
            raise ValueError(f"Frame {frame_idx}: unexpected EOF before BOX BOUNDS")
        if not box_header.startswith("ITEM: BOX BOUNDS"):
            raise ValueError(
                f"Frame {frame_idx}: expected 'ITEM: BOX BOUNDS ...', got: "
                f"{box_header.strip()!r}"
            )
        for _ in range(3):
            if not fh.readline():
                raise ValueError(
                    f"Frame {frame_idx}: unexpected EOF while reading BOX BOUNDS values"
                )

        atoms_header = fh.readline()
        if not atoms_header:
            raise ValueError(f"Frame {frame_idx}: unexpected EOF before ATOMS header")
        col_map = self._parse_atoms_header(atoms_header, frame_idx)
        return n_atoms, col_map

    def _scan_metadata(self) -> Tuple[int, int]:
        """
        First pass: count frames and validate a consistent atom count.
        """
        total_frames = 0
        n_atoms_ref: Optional[int] = None

        with self.dump_path.open("r", encoding="utf-8") as fh:
            frame_idx = 0
            while True:
                preamble = self._read_frame_preamble(fh, frame_idx)
                if preamble is None:
                    break
                n_atoms, _ = preamble

                if n_atoms_ref is None:
                    n_atoms_ref = n_atoms
                elif n_atoms != n_atoms_ref:
                    raise ValueError(
                        f"Frame {frame_idx}: NUMBER OF ATOMS changed from "
                        f"{n_atoms_ref} to {n_atoms}"
                    )

                for _ in range(n_atoms):
                    if not fh.readline():
                        raise ValueError(
                            f"Frame {frame_idx}: unexpected EOF within ATOMS section"
                        )

                total_frames += 1
                frame_idx += 1

        if total_frames == 0 or n_atoms_ref is None:
            raise ValueError("No frames found in dump file")
        return total_frames, n_atoms_ref

    def _read_n_lines(
        self, fh, n_lines: int, frame_idx: int, context: str
    ) -> list[str]:
        """Read exactly n_lines and raise a clear error on premature EOF."""
        lines = []
        for _ in range(n_lines):
            line = fh.readline()
            if not line:
                raise ValueError(
                    f"Frame {frame_idx}: unexpected EOF while reading {context}"
                )
            lines.append(line)
        return lines

    @staticmethod
    def _parse_lines_to_array(
        lines: list[str], n_cols: int, frame_idx: int
    ) -> np.ndarray:
        """Robust fallback parser for ATOMS lines."""
        out = np.empty((len(lines), n_cols), dtype=np.float64)
        for i, line in enumerate(lines):
            parts = line.split()
            if len(parts) != n_cols:
                raise ValueError(
                    f"Frame {frame_idx}: expected {n_cols} columns in ATOMS line, "
                    f"got {len(parts)}: {line.strip()!r}"
                )
            out[i, :] = np.asarray(parts, dtype=np.float64)
        return out

    def _read_atoms_block(
        self, fh, n_atoms: int, n_cols: int, frame_idx: int
    ) -> np.ndarray:
        """
        Read one ATOMS block and return a numeric matrix of shape (n_atoms, n_cols).
        """
        lines = self._read_n_lines(fh, n_atoms, frame_idx, "ATOMS section")
        if not self.use_fromstring_parser:
            return self._parse_lines_to_array(lines, n_cols, frame_idx)

        block_text = "".join(lines)
        arr = np.fromstring(block_text, sep=" ", dtype=np.float64)
        expected = n_atoms * n_cols
        if arr.size != expected:
            # Fall back to strict line parser when `fromstring` cannot parse
            # exactly (e.g., unexpected non-numeric tokens).
            return self._parse_lines_to_array(lines, n_cols, frame_idx)
        return arr.reshape(n_atoms, n_cols)

    def read_velocities(
        self,
        skip_frames: int = 0,
        frame_interval: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Read velocities into shape (n_frames, n_atoms, 3), ordered by ascending atom id.

        Parameters
        ----------
        skip_frames
            Number of initial frames to drop.
        frame_interval
            Keep every `frame_interval`-th frame after skipping.

        Returns
        -------
        velocities, atom_ids, atom_types
            velocities: shape (n_kept_frames, n_atoms, 3)
            atom_ids: shape (n_atoms,), ascending
            atom_types: shape (n_atoms,), aligned with atom_ids
        """
        if skip_frames < 0:
            raise ValueError("skip_frames must be >= 0")
        if frame_interval <= 0:
            raise ValueError("frame_interval must be >= 1")
        if not self.dump_path.exists():
            raise FileNotFoundError(f"Dump file not found: {self.dump_path}")

        total_frames, n_atoms = self._scan_metadata()
        if skip_frames >= total_frames:
            raise ValueError(
                f"skip_frames={skip_frames} leaves no frames "
                f"(total_frames={total_frames})"
            )

        n_kept = (total_frames - skip_frames + frame_interval - 1) // frame_interval
        if n_kept < 1:
            raise ValueError("No frames selected by skip_frames/frame_interval")

        if self.use_memmap:
            target_dir = self.memmap_dir if self.memmap_dir is not None else Path(tempfile.gettempdir())
            fd, mm_path = tempfile.mkstemp(
                prefix="vdos_vel_",
                suffix=".dat",
                dir=str(target_dir),
            )
            os.close(fd)
            self._memmap_path = Path(mm_path)
            velocities = np.memmap(
                self._memmap_path,
                mode="w+",
                dtype=self.dtype,
                shape=(n_kept, n_atoms, 3),
            )
        else:
            velocities = np.empty((n_kept, n_atoms, 3), dtype=self.dtype)

        atom_ids: Optional[np.ndarray] = None
        atom_types: Optional[np.ndarray] = None

        with self.dump_path.open("r", encoding="utf-8") as fh:
            frame_idx = 0
            kept_idx = 0
            while True:
                preamble = self._read_frame_preamble(fh, frame_idx)
                if preamble is None:
                    break
                n_atoms_frame, col_map = preamble
                if n_atoms_frame != n_atoms:
                    raise ValueError(
                        f"Frame {frame_idx}: NUMBER OF ATOMS mismatch (expected {n_atoms}, "
                        f"found {n_atoms_frame})"
                    )

                keep = frame_idx >= skip_frames and (
                    (frame_idx - skip_frames) % frame_interval == 0
                )

                col_id = col_map["id"]
                col_type = col_map["type"]
                col_vx = col_map["vx"]
                col_vy = col_map["vy"]
                col_vz = col_map["vz"]
                n_cols = len(col_map)

                if keep:
                    block = self._read_atoms_block(
                        fh=fh,
                        n_atoms=n_atoms,
                        n_cols=n_cols,
                        frame_idx=frame_idx,
                    )
                    ids_raw = block[:, col_id].astype(np.int64, copy=False)
                    types_raw = block[:, col_type].astype(np.int32, copy=False)
                    vel_raw = block[:, (col_vx, col_vy, col_vz)]

                    if atom_ids is None:
                        if self.assume_sorted_ids:
                            if np.any(ids_raw[1:] <= ids_raw[:-1]):
                                raise ValueError(
                                    f"Frame {frame_idx}: assume_sorted_ids=True requires "
                                    "strictly increasing atom ids in each frame. "
                                    "Use `dump_modify ... sort id` in LAMMPS or disable "
                                    "assume_sorted_ids."
                                )
                            atom_ids = ids_raw.copy()
                            atom_types = types_raw.copy()
                            velocities[kept_idx] = vel_raw.astype(
                                self.dtype, copy=False
                            )
                        else:
                            order = np.argsort(ids_raw, kind="mergesort")
                            ids_sorted = ids_raw[order]
                            if np.any(np.diff(ids_sorted) == 0):
                                raise ValueError(
                                    f"Frame {frame_idx}: duplicate atom IDs detected"
                                )
                            atom_ids = ids_sorted
                            atom_types = types_raw[order]
                            velocities[kept_idx] = vel_raw[order].astype(
                                self.dtype, copy=False
                            )
                    else:
                        assert atom_types is not None
                        if self.assume_sorted_ids:
                            velocities[kept_idx] = vel_raw.astype(self.dtype, copy=False)
                        else:
                            assert atom_ids is not None
                            pos = np.searchsorted(atom_ids, ids_raw)
                            if np.any(pos < 0) or np.any(pos >= n_atoms):
                                raise ValueError(
                                    f"Frame {frame_idx}: atom ids out of reference range"
                                )
                            if np.any(atom_ids[pos] != ids_raw):
                                bad_idx = np.where(atom_ids[pos] != ids_raw)[0][:5]
                                bad_ids = ids_raw[bad_idx].tolist()
                                raise ValueError(
                                    f"Frame {frame_idx}: unknown atom ids detected: {bad_ids}"
                                )

                            # Duplicate IDs imply missing IDs too for fixed atom count.
                            pos_sorted = np.sort(pos)
                            if np.any(np.diff(pos_sorted) == 0):
                                raise ValueError(
                                    f"Frame {frame_idx}: duplicate atom IDs detected"
                                )
                            if pos_sorted[0] != 0 or pos_sorted[-1] != n_atoms - 1:
                                raise ValueError(
                                    f"Frame {frame_idx}: atom ID coverage is incomplete"
                                )

                            if np.any(types_raw != atom_types[pos]):
                                bad_idx = np.where(types_raw != atom_types[pos])[0][0]
                                bad_id = int(ids_raw[bad_idx])
                                prev_t = int(atom_types[pos[bad_idx]])
                                curr_t = int(types_raw[bad_idx])
                                raise ValueError(
                                    f"Frame {frame_idx}: atom type changed for id "
                                    f"{bad_id} ({prev_t} -> {curr_t})"
                                )

                            frame_vel = np.empty((n_atoms, 3), dtype=np.float64)
                            frame_vel[pos, :] = vel_raw
                            velocities[kept_idx] = frame_vel.astype(self.dtype, copy=False)

                    kept_idx += 1
                else:
                    self._read_n_lines(fh, n_atoms, frame_idx, "skipped ATOMS section")

                frame_idx += 1

        if kept_idx != n_kept:
            velocities = velocities[:kept_idx]

        if atom_ids is None or atom_types is None:
            raise ValueError("No frame retained after applying skip/frame_interval")

        return velocities, atom_ids, atom_types


class VDOSCalculator:
    """
    Compute per-atom DOS from mass-weighted VACF using FFT-based autocorrelation.
    """

    def __init__(
        self,
        dt_fs: float,
        Nc: int,
        omega_max: float,
        num_dos_points: Optional[int],
        masses_by_type: Mapping[int, float],
        remove_com: bool = False,
        vacf_normalization: Literal["unbiased", "biased"] = "unbiased",
        gpumd_strict: bool = True,
        chunk_size_atoms: int = 128,
        freq_block_size: int = 256,
    ) -> None:
        """
        Parameters
        ----------
        dt_fs
            Effective frame spacing in fs used for VACF (already includes dump_stride
            and sample_interval if applicable).
        Nc
            Correlation length (number of lags from 0 to Nc-1).
        omega_max
            Maximum angular frequency grid value in THz (GPUMD's omega).
        num_dos_points
            Number of DOS points on [0, omega_max]. If None, defaults to Nc.
        masses_by_type
            Mapping from LAMMPS atom type to mass in amu.
        remove_com
            If True, subtract mass-weighted COM velocity from each frame.
        vacf_normalization
            VACF averaging convention:
            - "unbiased": divide lag τ by (T-τ) (default)
            - "biased": divide lag τ by T
        gpumd_strict
            If True, enforce GPUMD-style conventions:
            - omega is angular frequency and transform uses cos(omega*t)
            - per-group DOS normalization follows
              integral[(DOSx+DOSy+DOSz) d omega / (2*pi)] = 3*N_group
              (here N_group=1 for per-atom output)
        chunk_size_atoms
            Atom chunk size used during FFT processing.
        freq_block_size
            Frequency block size used during cosine transform.
        """
        if dt_fs <= 0.0:
            raise ValueError("dt_fs must be > 0")
        if Nc <= 0:
            raise ValueError("Nc must be > 0")
        if omega_max <= 0.0:
            raise ValueError("omega_max must be > 0")
        if num_dos_points is not None and num_dos_points <= 1:
            raise ValueError("num_dos_points must be >= 2")
        if chunk_size_atoms <= 0:
            raise ValueError("chunk_size_atoms must be >= 1")
        if freq_block_size <= 0:
            raise ValueError("freq_block_size must be >= 1")
        if len(masses_by_type) == 0:
            raise ValueError("masses_by_type must not be empty")
        if vacf_normalization not in ("unbiased", "biased"):
            raise ValueError("vacf_normalization must be 'unbiased' or 'biased'")

        self.dt_fs = float(dt_fs)
        self.dt_ps = self.dt_fs * 1.0e-3
        self.Nc = int(Nc)
        self.omega_max = float(omega_max)
        self.num_dos_points = int(num_dos_points) if num_dos_points is not None else int(Nc)
        self.masses_by_type = {int(k): float(v) for k, v in masses_by_type.items()}
        self.remove_com = bool(remove_com)
        self.vacf_normalization = vacf_normalization
        self.gpumd_strict = bool(gpumd_strict)
        self.chunk_size_atoms = int(chunk_size_atoms)
        self.freq_block_size = int(freq_block_size)

        self.total_mvac_xyz: Optional[np.ndarray] = None

    def _mass_array_from_types(self, atom_types: np.ndarray) -> np.ndarray:
        """Build a per-atom mass array from atom types."""
        unknown = sorted({int(t) for t in atom_types.tolist()} - set(self.masses_by_type))
        if unknown:
            raise ValueError(
                f"masses_by_type is missing atom types: {unknown}. "
                "Add all types used in the dump."
            )
        masses = np.array(
            [self.masses_by_type[int(t)] for t in atom_types.tolist()],
            dtype=np.float64,
        )
        if np.any(masses <= 0.0):
            raise ValueError("All masses must be positive")
        return masses

    def _autocorr_fft(
        self, signals: np.ndarray, n_lags: int, nfft: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute autocorrelation for many 1D signals with FFT.

        Parameters
        ----------
        signals
            Shape (T, M), where T is number of frames and M is number of signals.
        n_lags
            Number of lags to return from 0..n_lags-1.

        Returns
        -------
        acf
            Shape (M, n_lags).
        """
        if signals.ndim != 2:
            raise ValueError("signals must have shape (T, M)")
        n_frames, _ = signals.shape
        if n_lags > n_frames:
            raise ValueError(
                f"n_lags={n_lags} exceeds available frames {n_frames}"
            )

        if nfft is None:
            nfft = 1 << (2 * n_frames - 1).bit_length()
        spectrum = np.fft.rfft(signals, n=nfft, axis=0)
        power = spectrum * np.conjugate(spectrum)
        acf_full = np.fft.irfft(power, n=nfft, axis=0)[:n_lags, :]

        if self.vacf_normalization == "unbiased":
            denom = (n_frames - np.arange(n_lags, dtype=np.float64))[:, None]
        else:
            denom = float(n_frames)
        acf_full = acf_full / denom
        return acf_full.T

    def _autocorr_sum_fft(
        self, signals: np.ndarray, n_lags: int, nfft: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute summed autocorrelation over many signals with one inverse FFT.

        Returns
        -------
        acf_sum
            Shape (n_lags,), equals sum_m ACF(signals[:, m]).
        """
        if signals.ndim != 2:
            raise ValueError("signals must have shape (T, M)")
        n_frames, _ = signals.shape
        if n_lags > n_frames:
            raise ValueError(
                f"n_lags={n_lags} exceeds available frames {n_frames}"
            )

        if nfft is None:
            nfft = 1 << (2 * n_frames - 1).bit_length()
        spectrum = np.fft.rfft(signals, n=nfft, axis=0)
        power_sum = np.sum(spectrum * np.conjugate(spectrum), axis=1)
        acf_sum = np.fft.irfft(power_sum, n=nfft)[:n_lags]

        if self.vacf_normalization == "unbiased":
            denom = n_frames - np.arange(n_lags, dtype=np.float64)
        else:
            denom = float(n_frames)
        return acf_sum / denom

    def _build_cos_blocks(self, omega: np.ndarray) -> list[np.ndarray]:
        """
        Build cosine-transform blocks to avoid recomputing cos(omega*tau) repeatedly.
        """
        tau_ps = np.arange(self.Nc, dtype=np.float64) * self.dt_ps
        cos_blocks: list[np.ndarray] = []
        for w0 in range(0, omega.size, self.freq_block_size):
            w1 = min(w0 + self.freq_block_size, omega.size)
            omega_block = omega[w0:w1]
            cos_blocks.append(np.cos(np.outer(omega_block, tau_ps)))
        return cos_blocks

    def _vacf_to_dos(
        self,
        vacf: np.ndarray,
        omega: np.ndarray,
        cos_blocks: Optional[list[np.ndarray]] = None,
    ) -> np.ndarray:
        """
        Convert VACF to DOS via discrete cosine transform on an omega grid.

        Notes
        -----
        In GPUMD strict mode, omega is angular frequency and the transform uses
        cos(omega * t), where omega is the first column of dos.out.
        """
        if vacf.ndim != 2:
            raise ValueError("vacf must have shape (n_signals, Nc)")
        if vacf.shape[1] != self.Nc:
            raise ValueError(f"vacf second axis must be Nc={self.Nc}")

        if cos_blocks is None:
            cos_blocks = self._build_cos_blocks(omega)
        dos = np.empty((vacf.shape[0], omega.size), dtype=np.float64)

        w0 = 0
        for cos_matrix in cos_blocks:
            w1 = w0 + cos_matrix.shape[0]
            dos[:, w0:w1] = 2.0 * self.dt_ps * (vacf @ cos_matrix.T)
            w0 = w1
        return dos

    def compute_dos(
        self,
        velocities: np.ndarray,
        atom_types: np.ndarray,
        return_per_atom: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Compute DOS components (x,y,z) with optional per-atom storage.

        Parameters
        ----------
        velocities
            Array of shape (n_frames, n_atoms, 3).
        atom_types
            Array of shape (n_atoms,), aligned with the second axis of velocities.
        return_per_atom
            If True, also return per-atom DOS tensor.

        Returns
        -------
        omega, dos_group_xyz, dos_atom_xyz
            omega: shape (num_dos_points,)
            dos_group_xyz: shape (3, num_dos_points), one-group DOS of all atoms
            dos_atom_xyz: shape (n_atoms, 3, num_dos_points) or None
        """
        if velocities.ndim != 3 or velocities.shape[2] != 3:
            raise ValueError("velocities must have shape (n_frames, n_atoms, 3)")
        n_frames, n_atoms, _ = velocities.shape
        if atom_types.ndim != 1 or atom_types.shape[0] != n_atoms:
            raise ValueError(
                "atom_types must have shape (n_atoms,) and align with velocities"
            )
        if n_frames < self.Nc:
            raise ValueError(
                f"Nc={self.Nc} exceeds sampled frame count ({n_frames}). "
                "Reduce Nc or use more frames."
            )

        masses = self._mass_array_from_types(atom_types)
        omega = np.linspace(
            0.0, self.omega_max, self.num_dos_points, dtype=np.float64
        )
        cos_blocks = self._build_cos_blocks(omega)
        nfft = 1 << (2 * n_frames - 1).bit_length()
        dos_group_xyz = np.zeros((3, self.num_dos_points), dtype=np.float64)
        dos_atom_xyz: Optional[np.ndarray]
        if return_per_atom:
            dos_atom_xyz = np.empty((n_atoms, 3, self.num_dos_points), dtype=np.float64)
        else:
            dos_atom_xyz = None
        total_mvac_xyz = np.zeros((self.Nc, 3), dtype=np.float64)

        if self.remove_com:
            mass_weights = masses / masses.sum()
            v_cm = np.tensordot(
                velocities.astype(np.float64, copy=False),
                mass_weights,
                axes=(1, 0),
            )  # (n_frames, 3)
        else:
            v_cm = None

        for i0 in range(0, n_atoms, self.chunk_size_atoms):
            i1 = min(i0 + self.chunk_size_atoms, n_atoms)
            mass_sqrt = np.sqrt(masses[i0:i1])[None, :, None]
            v_chunk = velocities[:, i0:i1, :].astype(np.float64, copy=False)
            if self.remove_com:
                assert v_cm is not None
                s_chunk = (v_chunk - v_cm[:, None, :]) * mass_sqrt
            else:
                s_chunk = v_chunk * mass_sqrt

            if not return_per_atom:
                for comp in range(3):
                    signals = s_chunk[:, :, comp]  # (n_frames, chunk_atoms)
                    vacf_sum = self._autocorr_sum_fft(
                        signals,
                        self.Nc,
                        nfft=nfft,
                    )  # (Nc,)
                    total_mvac_xyz[:, comp] += vacf_sum
                continue

            dos_chunk_xyz = np.empty((i1 - i0, 3, self.num_dos_points), dtype=np.float64)
            for comp in range(3):
                signals = s_chunk[:, :, comp]  # (n_frames, chunk_atoms)
                vacf = self._autocorr_fft(signals, self.Nc, nfft=nfft)  # (chunk_atoms, Nc)
                total_mvac_xyz[:, comp] += vacf.sum(axis=0)

                dos_comp = self._vacf_to_dos(
                    vacf,
                    omega,
                    cos_blocks=cos_blocks,
                )  # (chunk_atoms, nw)
                dos_chunk_xyz[:, comp, :] = dos_comp

            total_area_omega = _trapz(np.sum(dos_chunk_xyz, axis=1), omega, axis=1)
            if np.any(np.isclose(total_area_omega, 0.0, atol=1e-20)):
                raise ValueError(
                    "Encountered near-zero total DOS integral during normalization. "
                    "Try larger Nc, more frames, or check trajectory quality."
                )

            if self.gpumd_strict:
                # GPUMD normalization convention for each group:
                # integral[(DOSx + DOSy + DOSz) d omega / (2*pi)] = 3 * N_group.
                # Here each atom is one group, so N_group=1 -> target integral over omega is 6*pi.
                target_area_omega = 6.0 * np.pi
            else:
                target_area_omega = 3.0

            dos_chunk_xyz *= (target_area_omega / total_area_omega)[:, None, None]
            if dos_atom_xyz is not None:
                dos_atom_xyz[i0:i1] = dos_chunk_xyz

        # Group DOS is always produced from the all-atom VACF (GPUMD group semantics).
        dos_group_xyz = self._vacf_to_dos(
            total_mvac_xyz.T,
            omega,
            cos_blocks=cos_blocks,
        )
        group_area_omega = _trapz(np.sum(dos_group_xyz, axis=0), omega)
        if np.isclose(group_area_omega, 0.0, atol=1e-20):
            raise ValueError(
                "Encountered near-zero group DOS integral during normalization. "
                "Try larger Nc, more frames, or check trajectory quality."
            )
        if self.gpumd_strict:
            target_group_area_omega = 6.0 * np.pi * float(n_atoms)
        else:
            target_group_area_omega = 3.0 * float(n_atoms)
        dos_group_xyz *= target_group_area_omega / group_area_omega

        # mvac.out in GPUMD is mass-normalized VAC: divide weighted sum by total mass.
        self.total_mvac_xyz = total_mvac_xyz / float(np.sum(masses))
        return omega, dos_group_xyz, dos_atom_xyz

    def compute_group_dos(
        self, velocities: np.ndarray, atom_types: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute DOS for one group containing all atoms.

        Returns
        -------
        omega, dos_group_xyz
            omega: shape (num_dos_points,)
            dos_group_xyz: shape (3, num_dos_points)
        """
        omega, dos_group_xyz, _ = self.compute_dos(
            velocities, atom_types, return_per_atom=False
        )
        return omega, dos_group_xyz

    def compute_per_atom_dos(
        self, velocities: np.ndarray, atom_types: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute per-atom DOS components (x,y,z).
        """
        omega, _, dos_atom_xyz = self.compute_dos(
            velocities, atom_types, return_per_atom=True
        )
        assert dos_atom_xyz is not None
        return omega, dos_atom_xyz

    def write_group_dos_out(
        self, path: str | Path, omega: np.ndarray, dos_group_xyz: np.ndarray
    ) -> None:
        """
        Write GPUMD-like dos.out for a single group (all atoms).
        Columns: omega DOSx DOSy DOSz
        """
        if omega.ndim != 1:
            raise ValueError("omega must be 1D")
        if dos_group_xyz.ndim != 2 or dos_group_xyz.shape[0] != 3:
            raise ValueError("dos_group_xyz must have shape (3, num_dos_points)")
        if dos_group_xyz.shape[1] != omega.size:
            raise ValueError("omega size must match dos_group_xyz second axis")

        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        data = np.column_stack((omega, dos_group_xyz[0], dos_group_xyz[1], dos_group_xyz[2]))
        np.savetxt(out_path, data, fmt="%.10e")

    def write_dos_out(
        self,
        path: str | Path,
        omega: np.ndarray,
        dos_xyz: np.ndarray,
        atom_ids: np.ndarray,
        format: str = "gpumd_concat",
    ) -> None:
        """
        Write per-atom DOS in GPUMD concatenated-group style.

        For each atom (group size 1), write `num_dos_points` lines:
        omega DOSx DOSy DOSz
        """
        if format != "gpumd_concat":
            raise ValueError("Only format='gpumd_concat' is supported")
        if dos_xyz.ndim != 3 or dos_xyz.shape[1] != 3:
            raise ValueError("dos_xyz must have shape (n_atoms, 3, num_dos_points)")
        if atom_ids.ndim != 1 or atom_ids.shape[0] != dos_xyz.shape[0]:
            raise ValueError("atom_ids must align with dos_xyz first axis")
        if dos_xyz.shape[2] != omega.size:
            raise ValueError("omega size must match dos_xyz third axis")

        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for i in range(dos_xyz.shape[0]):
                block = np.column_stack(
                    (omega, dos_xyz[i, 0, :], dos_xyz[i, 1, :], dos_xyz[i, 2, :])
                )
                np.savetxt(f, block, fmt="%.10e")

    def write_mvac_out(
        self,
        path: str | Path,
        tau_ps: np.ndarray,
        mvac_xyz: np.ndarray,
    ) -> None:
        """
        Write mass-normalized VACF as:
        tau_ps mvac_x mvac_y mvac_z
        """
        if mvac_xyz.ndim != 2 or mvac_xyz.shape[1] != 3:
            raise ValueError("mvac_xyz must have shape (Nc, 3)")
        if tau_ps.ndim != 1 or tau_ps.shape[0] != mvac_xyz.shape[0]:
            raise ValueError("tau_ps length must match mvac_xyz")

        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        data = np.column_stack((tau_ps, mvac_xyz))
        np.savetxt(out_path, data, fmt="%.10e")


class PPRCalculator:
    """Compute and write PPR from per-atom PDOS."""

    def compute_ppr(self, omega: np.ndarray, dos_xyz: np.ndarray) -> np.ndarray:
        """
        Compute PPR(omega) using per-atom scalar PDOS_i = DOSx + DOSy + DOSz:

        PPR = (1/N) * ((sum_i PDOS_i^2)^2 / sum_i PDOS_i^4)
        """
        if omega.ndim != 1:
            raise ValueError("omega must be 1D")
        if dos_xyz.ndim != 3 or dos_xyz.shape[1] != 3:
            raise ValueError("dos_xyz must have shape (n_atoms, 3, num_dos_points)")
        if dos_xyz.shape[2] != omega.size:
            raise ValueError("omega size must match dos_xyz third axis")

        n_atoms = dos_xyz.shape[0]
        pdos = dos_xyz[:, 0, :] + dos_xyz[:, 1, :] + dos_xyz[:, 2, :]
        sum2 = np.sum(pdos * pdos, axis=0)
        sum4 = np.sum(pdos * pdos * pdos * pdos, axis=0)

        ppr = np.zeros_like(omega, dtype=np.float64)
        mask = sum4 > 0.0
        ppr[mask] = (sum2[mask] ** 2) / (sum4[mask] * float(n_atoms))
        return ppr

    def write_ppr(self, path: str | Path, omega: np.ndarray, ppr: np.ndarray) -> None:
        """Write ppr.out with two columns: omega PPR."""
        if omega.ndim != 1 or ppr.ndim != 1 or omega.size != ppr.size:
            raise ValueError("omega and ppr must be 1D arrays with equal size")
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(out_path, np.column_stack((omega, ppr)), fmt="%.10e")


@dataclass
class RunConfig:
    """User-editable execution settings."""

    test_mode: bool = False

    dump_path: str = "dump.lammpstrj"
    dt_fs: float = 1.0
    dump_stride: int = 1
    sample_interval: int = 1
    Nc: int = 4096
    omega_max: float = 400.0
    num_dos_points: Optional[int] = None
    masses_by_type: Dict[int, float] = field(default_factory=lambda: {1: 28.0855})
    skip_frames: int = 0
    remove_com: bool = True
    out_dir: str = "vdos_output"
    gpumd_strict: bool = True
    vacf_normalization: Literal["unbiased", "biased"] = "unbiased"

    velocity_dtype: np.dtype | type = np.float32
    use_memmap: bool = False
    memmap_dir: Optional[str] = None
    assume_sorted_ids: bool = False
    use_fromstring_parser: bool = True
    chunk_size_atoms: int = 128
    freq_block_size: int = 256

    write_dos: bool = True
    write_mvac: bool = True
    write_dos_atom: bool = False
    compute_ppr: bool = False

    assert_normalization: bool = True
    write_timing_json: bool = True


def run_vdos_pipeline(config: RunConfig) -> None:
    """Execute VDOS/PDOS workflow using the provided configuration."""
    t_total = time.perf_counter()
    timings: Dict[str, float] = {}
    out_path = Path(config.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    reader = LammpsDumpReader(
        dump_path=config.dump_path,
        dtype=config.velocity_dtype,
        use_memmap=config.use_memmap,
        memmap_dir=config.memmap_dir,
        assume_sorted_ids=config.assume_sorted_ids,
        use_fromstring_parser=config.use_fromstring_parser,
    )
    velocities: Optional[np.ndarray] = None
    generated_files = []

    try:
        t0 = time.perf_counter()
        velocities, atom_ids, atom_types = reader.read_velocities(
            skip_frames=config.skip_frames,
            frame_interval=config.sample_interval,
        )
        timings["read_dump_s"] = time.perf_counter() - t0

        if velocities.shape[0] < 2:
            raise ValueError(
                f"Too few sampled frames ({velocities.shape[0]}). Increase trajectory length."
            )

        effective_dt_fs = (
            config.dt_fs * float(config.dump_stride) * float(config.sample_interval)
        )
        need_per_atom_dos = bool(config.write_dos_atom or config.compute_ppr)

        calc = VDOSCalculator(
            dt_fs=effective_dt_fs,
            Nc=config.Nc,
            omega_max=config.omega_max,
            num_dos_points=config.num_dos_points,
            masses_by_type=config.masses_by_type,
            remove_com=config.remove_com,
            vacf_normalization=config.vacf_normalization,
            gpumd_strict=config.gpumd_strict,
            chunk_size_atoms=config.chunk_size_atoms,
            freq_block_size=config.freq_block_size,
        )

        t0 = time.perf_counter()
        omega, dos_group_xyz, dos_atom_xyz = calc.compute_dos(
            velocities,
            atom_types,
            return_per_atom=need_per_atom_dos,
        )
        timings["compute_dos_s"] = time.perf_counter() - t0

        if config.assert_normalization:
            t0 = time.perf_counter()
            group_integral = _trapz(np.sum(dos_group_xyz, axis=0), omega)
            if config.gpumd_strict:
                group_integral = group_integral / (2.0 * np.pi)
                target = 3.0 * float(atom_ids.size)
                rule = "integral(sum_xyz_dos * d_omega / (2*pi))"
            else:
                target = 3.0 * float(atom_ids.size)
                rule = "integral(sum_xyz_dos * d_omega)"

            atol = max(1.0e-2, 5.0e-2 * target)
            if not np.isclose(group_integral, target, atol=atol, rtol=5.0e-2):
                raise AssertionError(
                    "DOS normalization check failed: "
                    f"{rule} should be ~{target:.6f}, got {group_integral:.6f}"
                )
            timings["check_normalization_s"] = time.perf_counter() - t0

        if config.write_dos:
            t0 = time.perf_counter()
            dos_path = out_path / "dos.out"
            calc.write_group_dos_out(dos_path, omega, dos_group_xyz)
            timings["write_dos_s"] = time.perf_counter() - t0
            generated_files.append(str(dos_path))

        if config.write_dos_atom:
            if dos_atom_xyz is None:
                raise RuntimeError(
                    "Internal error: per-atom DOS was not computed but write_dos_atom=True."
                )
            t0 = time.perf_counter()
            dos_atom_path = out_path / "dos_atom.out"
            calc.write_dos_out(
                dos_atom_path,
                omega,
                dos_atom_xyz,
                atom_ids,
                format="gpumd_concat",
            )
            timings["write_dos_atom_s"] = time.perf_counter() - t0
            generated_files.append(str(dos_atom_path))

        if config.compute_ppr:
            if dos_atom_xyz is None:
                raise RuntimeError(
                    "Internal error: per-atom DOS was not computed but compute_ppr=True."
                )
            ppr_calc = PPRCalculator()
            t0 = time.perf_counter()
            ppr = ppr_calc.compute_ppr(omega, dos_atom_xyz)
            timings["compute_ppr_s"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            ppr_path = out_path / "ppr.out"
            ppr_calc.write_ppr(ppr_path, omega, ppr)
            timings["write_ppr_s"] = time.perf_counter() - t0
            generated_files.append(str(ppr_path))

        if config.write_mvac and calc.total_mvac_xyz is not None:
            t0 = time.perf_counter()
            tau_ps = np.arange(calc.Nc, dtype=np.float64) * calc.dt_ps
            mvac_path = out_path / "mvac.out"
            calc.write_mvac_out(mvac_path, tau_ps, calc.total_mvac_xyz)
            timings["write_mvac_s"] = time.perf_counter() - t0
            generated_files.append(str(mvac_path))

        timings["total_s"] = time.perf_counter() - t_total
        meta = {
            "implementation_version": "v2",
            "source_dump": str(config.dump_path),
            "atom_ids": atom_ids.tolist(),
            "num_atoms": int(atom_ids.size),
            "num_frames_used": int(velocities.shape[0]),
            "dt_fs_input": float(config.dt_fs),
            "dump_stride": int(config.dump_stride),
            "sample_interval": int(config.sample_interval),
            "dt_fs_effective": float(effective_dt_fs),
            "Nc": int(config.Nc),
            "omega_max": float(config.omega_max),
            "omega_max_linear_thz": float(config.omega_max / (2.0 * np.pi)),
            "num_dos_points": int(omega.size),
            "remove_com": bool(config.remove_com),
            "gpumd_strict": bool(calc.gpumd_strict),
            "omega_definition": "angular_frequency_omega_in_THz",
            "dos_normalization_rule": (
                "integral(sum_xyz_dos * d_omega / (2*pi)) = 3 per atom-group"
                if config.gpumd_strict
                else "integral(sum_xyz_dos * d_omega) = 3 per atom-group"
            ),
            "vacf_normalization": calc.vacf_normalization,
            "output_options": {
                "write_dos": bool(config.write_dos),
                "write_mvac": bool(config.write_mvac),
                "write_dos_atom": bool(config.write_dos_atom),
                "compute_ppr": bool(config.compute_ppr),
            },
            "reader_options": {
                "assume_sorted_ids": bool(config.assume_sorted_ids),
                "use_fromstring_parser": bool(config.use_fromstring_parser),
                "use_memmap": bool(config.use_memmap),
            },
            "masses_by_type": {
                str(k): float(v) for k, v in config.masses_by_type.items()
            },
            "timings_seconds": timings,
        }
        meta_path = out_path / "dos.meta.json"
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        generated_files.append(str(meta_path))

        if config.write_timing_json:
            timing_path = out_path / "timings.json"
            with timing_path.open("w", encoding="utf-8") as f:
                json.dump(timings, f, indent=2)
            generated_files.append(str(timing_path))

        for fp in generated_files:
            print(f"Wrote: {fp}")
        print("Timing summary [s]:")
        for k, v in timings.items():
            print(f"  {k}: {v:.6f}")
    finally:
        if isinstance(velocities, np.memmap):
            velocities.flush()
        reader.cleanup()


def _run_synthetic_test() -> None:
    """
    Minimal self-test:
    - Generates sinusoidal velocity signals.
    - Verifies DOS peak appears near target angular frequency.
    - Verifies GPUMD-style normalization:
      integral[(DOSx + DOSy + DOSz) d omega / (2*pi)] ~= 3 per atom.
    """
    rng = np.random.default_rng(0)

    n_frames = 4096
    n_atoms = 8
    dt_fs = 1.0
    target_nu_thz = 8.0
    target_omega_thz = 2.0 * np.pi * target_nu_thz
    omega_max = 2.0 * np.pi * 40.0
    Nc = 1024
    num_dos_points = 1024

    t_ps = np.arange(n_frames, dtype=np.float64) * (dt_fs * 1.0e-3)
    velocities = np.zeros((n_frames, n_atoms, 3), dtype=np.float32)
    for i in range(n_atoms):
        phase = 0.19 * i
        velocities[:, i, 0] = np.sin(2.0 * np.pi * target_nu_thz * t_ps + phase)
        velocities[:, i, 1] = np.sin(2.0 * np.pi * 12.0 * t_ps + phase)
        velocities[:, i, 2] = np.sin(2.0 * np.pi * 4.0 * t_ps + phase)

    velocities += 0.02 * rng.standard_normal(size=velocities.shape).astype(np.float32)
    atom_types = np.ones(n_atoms, dtype=np.int32)

    calc = VDOSCalculator(
        dt_fs=dt_fs,
        Nc=Nc,
        omega_max=omega_max,
        num_dos_points=num_dos_points,
        masses_by_type={1: 28.0855},
        remove_com=False,
        vacf_normalization="unbiased",
        gpumd_strict=True,
        chunk_size_atoms=4,
        freq_block_size=256,
    )
    omega, dos_xyz = calc.compute_per_atom_dos(velocities, atom_types)

    peak_omega = omega[np.argmax(dos_xyz[0, 0, :])]
    bin_width = omega[1] - omega[0]
    assert abs(peak_omega - target_omega_thz) <= (2.0 * bin_width + 1.0), (
        f"Peak angular-frequency mismatch: expected near {target_omega_thz:.3f} THz, "
        f"got {peak_omega:.3f} THz"
    )

    sum_area = _trapz(np.sum(dos_xyz, axis=1), omega, axis=1) / (2.0 * np.pi)
    assert np.allclose(sum_area, 3.0, atol=1.5e-1, rtol=5e-2), (
        f"GPUMD normalization check failed. Min/Max integral over omega/(2pi): "
        f"{sum_area.min():.6f}/{sum_area.max():.6f}"
    )

    ppr_calc = PPRCalculator()
    ppr = ppr_calc.compute_ppr(omega, dos_xyz)
    assert ppr.shape == omega.shape

    print("Synthetic test passed.")
    print(
        "Target omega (x): "
        f"{target_omega_thz:.3f} THz (from nu={target_nu_thz:.3f} THz), "
        f"detected peak: {peak_omega:.3f} THz"
    )


if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # User-editable settings
    # -------------------------------------------------------------------------
    config = RunConfig(
        # ---- Mode ----
        # True: synthetic self-test only / False: run from dump trajectory
        test_mode=False,

        # ---- Input trajectory & sampling ----
        # LAMMPS dump path (must contain id type x y z vx vy vz)
        dump_path="dump.lammpstrj",
        # MD integration timestep [fs]
        dt_fs=1.0,
        # Dump write interval in MD steps
        dump_stride=1,
        # Additional subsampling interval used in DOS computation
        sample_interval=1,
        # Number of initial frames to discard
        skip_frames=0,

        # ---- Correlation / frequency grid ----
        # VACF correlation length (number of lags)
        Nc=4096,
        # Max angular frequency omega [THz] for dos.out col1
        omega_max=400.0,
        # Number of DOS grid points on [0, omega_max], None -> Nc
        num_dos_points=None,

        # ---- Physics / conventions ----
        # Atom masses [amu] keyed by LAMMPS type
        masses_by_type={1: 28.0855},
        # Remove mass-weighted COM drift before VACF
        remove_com=True,
        # GPUMD-style omega definition and normalization
        gpumd_strict=True,
        # VACF normalization: "unbiased" (T-tau) or "biased" (T)
        vacf_normalization="unbiased",

        # ---- Performance / memory ----
        # Velocity storage dtype in memory
        velocity_dtype=np.float32,
        # Use np.memmap for large trajectory handling
        use_memmap=False,
        # Temp directory for memmap file (None -> system temp)
        memmap_dir=None,
        # Fast path: assume all frames are already sorted by id.
        # Requires LAMMPS `dump_modify ... sort id`.
        assume_sorted_ids=False,
        # Parse ATOMS blocks with np.fromstring (faster than split/float loops).
        use_fromstring_parser=True,
        # Atom chunk size in FFT/VACF loop
        chunk_size_atoms=128,
        # Frequency block size in cosine transform
        freq_block_size=256,

        # ---- Outputs ----
        # Output directory
        out_dir="vdos_output",
        # Write group DOS (GPUMD-like dos.out)
        write_dos=True,
        # Write mass-normalized VACF xyz (mvac.out)
        write_mvac=True,
        # Write per-atom concatenated DOS (dos_atom.out)
        write_dos_atom=False,
        # Compute/write ppr.out from per-atom DOS
        compute_ppr=False,

        # ---- Validation / profiling ----
        # Enable normalization assertion
        assert_normalization=True,
        # Write timings.json
        write_timing_json=True,
    )
    # -------------------------------------------------------------------------

    if config.test_mode:
        _run_synthetic_test()
    else:
        run_vdos_pipeline(config)
