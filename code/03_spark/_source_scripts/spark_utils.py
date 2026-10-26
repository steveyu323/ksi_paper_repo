"""
spark_utils.py
==============
Utility functions for CellSAM-based cell segmentation, tracking,
SPARK puncta quantification, HDF5 storage, and figure generation.

Usage
-----
    from spark_utils import (
        load_tiff_as_grayscale, read_frame,
        run_cellsam_segmentation,
        filter_small_cells, compute_iou_matrix, merge_to_reference,
        interpolate_missing_masks,
        quantify_spark, normalise_spark,
        save_to_hdf5, load_spark_run,
        plot_spark_overlay, make_paper_figure,
    )
"""

import os
import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import tifffile
import h5py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from skimage import measure
from skimage.morphology import remove_small_objects, disk, white_tophat
from skimage.segmentation import find_boundaries
from scipy.ndimage import binary_dilation
from tqdm import tqdm

warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════════════════════
# 1. TIFF LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_tiff_as_grayscale(path, channel=None):
    """
    Load a TIFF movie into a (T, Y, X) float32 array.
    Uses tifffile for bit-depth-accurate reading (PIL silently downcasts).

    Parameters
    ----------
    path    : str
    channel : None | int | 'max' | 'sum'
        None  → auto-select first non-zero channel
        int   → explicit channel index
        'max' → channel with highest mean signal
        'sum' → sum all channels

    Returns
    -------
    arr  : np.ndarray  (T, Y, X) float32
    meta : dict        loader diagnostics including 'selected_channel'
    """
    raw = tifffile.imread(path)
    meta = {
        'path':        path,
        'raw_shape':   list(raw.shape),
        'raw_dtype':   str(raw.dtype),
        'channel_arg': channel,
    }

    # normalise axis order → (T, Y, X) or (T, Y, X, C)
    if raw.ndim == 4:
        if raw.shape[1] < raw.shape[2] and raw.shape[1] < raw.shape[3]:
            raw = np.moveaxis(raw, 1, -1)   # (T,C,Y,X) → (T,Y,X,C)
            meta['axis_reorder'] = 'TCYX→TYXC'
        meta['n_channels']    = raw.shape[3]
        meta['channel_means'] = [float(raw[:, :, :, c].mean()) for c in range(raw.shape[3])]
        meta['channel_maxes'] = [int(raw[:, :, :, c].max())    for c in range(raw.shape[3])]
        print(f'Multichannel TIFF: {raw.shape}  dtype={raw.dtype}')
        for c in range(raw.shape[3]):
            print(f'  channel {c}: mean={meta["channel_means"][c]:.1f}  '
                  f'max={meta["channel_maxes"][c]}')

    if raw.ndim == 3:
        arr  = raw.astype(np.float32)
        meta['mode'] = 'grayscale_direct'

    elif raw.ndim == 4:
        if channel == 'max':
            selected = int(np.argmax(meta['channel_means']))
        elif channel == 'sum':
            arr      = raw.astype(np.float32).sum(axis=3)
            meta['mode'] = f'sum_{raw.shape[3]}_channels'
            selected = None
        elif channel is None:
            nonzero  = [c for c in range(raw.shape[3]) if meta['channel_means'][c] > 0]
            selected = nonzero[0] if nonzero else 0
            if len(nonzero) > 1:
                selected = nonzero[int(np.argmax([meta['channel_means'][c] for c in nonzero]))]
            meta['auto_selected'] = selected
        elif isinstance(channel, int):
            if channel >= raw.shape[3]:
                raise ValueError(f'channel={channel} but only {raw.shape[3]} available')
            selected = channel
        else:
            raise ValueError(f'Unknown channel: {channel!r}')

        if selected is not None:
            arr  = raw[:, :, :, selected].astype(np.float32)
            meta['mode']             = f'channel_{selected}_of_{raw.shape[3]}'
            meta['selected_channel'] = selected
    else:
        raise ValueError(f'Unexpected shape: {raw.shape}')

    assert arr.ndim == 3, f'Expected (T,Y,X), got {arr.shape}'
    meta['output_shape'] = list(arr.shape)
    meta['output_dtype'] = str(arr.dtype)
    meta['output_min']   = float(arr.min())
    meta['output_max']   = float(arr.max())
    meta['output_mean']  = float(arr.mean())

    print(f'\nSelected: {meta["mode"]}  shape={arr.shape}  '
          f'min={meta["output_min"]:.0f}  max={meta["output_max"]:.0f}  '
          f'mean={meta["output_mean"]:.1f}')
    return arr, meta


def make_read_frame(path, meta):
    """
    Return a read_frame(t) callable that reads one frame and selects
    the correct channel, using the meta dict from load_tiff_as_grayscale.
    """
    _ch = meta.get('selected_channel')

    def read_frame(t):
        f = tifffile.imread(path, key=t).astype(np.float32)
        if f.ndim == 3 and _ch is not None:
            return f[:, :, _ch]
        return f

    return read_frame


# ══════════════════════════════════════════════════════════════════════════════
# 2. CELLSAM SEGMENTATION
# ══════════════════════════════════════════════════════════════════════════════

def run_cellsam_segmentation(model, arr, clip_max=3000):
    """
    Run CellSAM segmentation on every frame of arr.

    Parameters
    ----------
    model    : loaded CellSAM model (eval + cuda)
    arr      : (T, Y, X) float32 raw fluorescence array
    clip_max : upper clip value before normalisation (default 3000)

    Returns
    -------
    masks_raw : list of (Y, X) int label arrays, one per frame
    """
    from model import process_and_segment_image   # local import — requires CellSAM env

    masks_raw = []
    for i in tqdm(range(arr.shape[0]), desc='CellSAM segmentation'):
        X_curr = arr[i, :, :].copy()
        X_curr[X_curr > clip_max] = clip_max
        X_norm = X_curr.astype(np.float32)
        img    = np.array([X_norm, X_norm, X_norm])
        img    = np.moveaxis(img, [0, 1, 2], [2, 0, 1])
        try:
            mask, _ = process_and_segment_image(model, img, normalize=True)
        except Exception:
            print(f'  no mask found at frame {i}')
            mask = np.zeros(X_curr.shape, dtype=np.int32)
        masks_raw.append(mask)
    return masks_raw


# ══════════════════════════════════════════════════════════════════════════════
# 3. MASK POST-PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def filter_small_cells(label_frame, min_size):
    """Remove labelled regions smaller than min_size px²."""
    out = label_frame.copy()
    for r in measure.regionprops(label_frame):
        if r.area < min_size:
            out[label_frame == r.label] = 0
    return out


def compute_iou_matrix(ref_labels, query_labels):
    """
    Compute IoU, IoR (intersection/ref area), and IoQ (intersection/query area)
    matrices between two label frames.

    Returns iou_mat, ior_mat, ioq_mat, ref_ids, query_ids
    """
    ref_ids   = np.unique(ref_labels);   ref_ids   = ref_ids[ref_ids != 0]
    query_ids = np.unique(query_labels); query_ids = query_ids[query_ids != 0]
    n_r, n_q  = len(ref_ids), len(query_ids)

    iou_mat = np.zeros((n_r, n_q), dtype=np.float32)
    ior_mat = np.zeros((n_r, n_q), dtype=np.float32)
    ioq_mat = np.zeros((n_r, n_q), dtype=np.float32)

    for i, rid in enumerate(ref_ids):
        ref_mask = (ref_labels == rid)
        ref_area = ref_mask.sum()
        for j, qid in enumerate(query_ids):
            q_mask       = (query_labels == qid)
            q_area       = q_mask.sum()
            intersection = (ref_mask & q_mask).sum()
            union        = (ref_mask | q_mask).sum()
            iou_mat[i, j] = intersection / union      if union    > 0 else 0
            ior_mat[i, j] = intersection / ref_area   if ref_area > 0 else 0
            ioq_mat[i, j] = intersection / q_area     if q_area   > 0 else 0

    return iou_mat, ior_mat, ioq_mat, ref_ids, query_ids


def merge_to_reference(ref_frame, query_frame, origin_frame,
                       iou_thresh=0.3, ioq_thresh=0.2, dilation_px=20,
                       soft_ior_thresh=0.15, soft_ioq_thresh=0.3,
                       recovery_ior_thresh=0.1, recovery_ioq_thresh=0.3,
                       verbose=False):
    """
    Four-stage merge of query_frame segments back to ref_frame cell IDs.

    Stage 1 — greedy IoR/IoQ match (descending confidence, no double-claim)
    Stage 2 — soft match for unassigned ref cells
    Stage 3 — origin recovery for cells absent from ref_frame
    Stage 4 — dilation absorption of unclaimed segments

    Parameters
    ----------
    ref_frame    : (Y, X) int — previous clean frame (rolling reference)
    query_frame  : (Y, X) int — current raw segmentation frame
    origin_frame : (Y, X) int — frozen frame-0 mask (cell ID registry)
    verbose      : print soft-match / recovery messages
    """
    iou_mat, ior_mat, ioq_mat, ref_ids, query_ids = compute_iou_matrix(
        ref_frame, query_frame)
    out     = np.zeros_like(query_frame)
    claimed = set()

    # ── stage 1: greedy match ────────────────────────────────────────────────
    candidates = []
    for i, rid in enumerate(ref_ids):
        for j, qid in enumerate(query_ids):
            score = max(ior_mat[i, j], ioq_mat[i, j])
            if score > 0:
                candidates.append((score, i, j, rid, qid))
    candidates.sort(key=lambda x: -x[0])

    assigned_refs = set()
    for score, i, j, rid, qid in candidates:
        if rid in assigned_refs or qid in claimed:
            continue
        if ior_mat[i, j] >= iou_thresh or ioq_mat[i, j] >= ioq_thresh:
            out[query_frame == qid] = rid
            claimed.add(qid)
            assigned_refs.add(rid)

    # ── stage 2: soft match ──────────────────────────────────────────────────
    for i, rid in enumerate(ref_ids):
        if rid in assigned_refs:
            continue
        avail = [(ior_mat[i, j], ioq_mat[i, j], j, qid)
                 for j, qid in enumerate(query_ids) if qid not in claimed]
        if not avail:
            continue
        best_ior, best_ioq, best_j, best_qid = max(avail, key=lambda x: x[0])
        if best_ior >= soft_ior_thresh or best_ioq >= soft_ioq_thresh:
            out[query_frame == best_qid] = rid
            claimed.add(best_qid)
            assigned_refs.add(rid)
            if verbose:
                print(f'  soft match ref {rid} → query {best_qid} '
                      f'(IoR={best_ior:.3f} IoQ={best_ioq:.3f})')
        else:
            if verbose:
                print(f'  ref {rid} unmatched (best IoR={best_ior:.3f})')

    # ── stage 3: origin recovery ─────────────────────────────────────────────
    origin_ids_arr = np.unique(origin_frame); origin_ids_arr = origin_ids_arr[origin_ids_arr != 0]
    missing_ids    = [oid for oid in origin_ids_arr if oid not in ref_ids]

    if missing_ids:
        _, ior_orig, ioq_orig, _, query_ids_o = compute_iou_matrix(
            origin_frame, query_frame)
        for oid in missing_ids:
            oi    = np.where(origin_ids_arr == oid)[0][0]
            avail = [(ior_orig[oi, j], ioq_orig[oi, j], j, qid)
                     for j, qid in enumerate(query_ids_o) if qid not in claimed]
            if not avail:
                continue
            best_ior, best_ioq, best_j, best_qid = max(avail, key=lambda x: x[0])
            if best_ior >= recovery_ior_thresh or best_ioq >= recovery_ioq_thresh:
                out[query_frame == best_qid] = oid
                claimed.add(best_qid)
                if verbose:
                    print(f'  recovered cell {oid} → query {best_qid} '
                          f'(IoR={best_ior:.3f} IoQ={best_ioq:.3f})')

    # ── stage 4: dilation absorption ─────────────────────────────────────────
    for qid in [q for q in query_ids if q not in claimed]:
        unclaimed_mask = (query_frame == qid)
        dilated        = binary_dilation(unclaimed_mask, iterations=dilation_px)
        neighbours     = out[dilated & ~unclaimed_mask]
        neighbours     = neighbours[neighbours > 0]
        if neighbours.size > 0:
            absorb_id = np.bincount(neighbours).argmax()
            out[unclaimed_mask] = absorb_id
            if verbose:
                print(f'  dilation: segment {qid} ({unclaimed_mask.sum()}px) '
                      f'→ absorbed into cell {absorb_id}')

    return out


def postprocess_masks(masks_raw, min_cell_size_px=5000,
                      iou_thresh=0.3, ioq_thresh=0.2, dilation_px=20,
                      verbose=False):
    """
    Apply size filtering and reference-based merge to a list of raw masks.

    Returns masks_clean (list of (Y,X) int arrays) and origin_frame.
    """
    origin_frame = None
    masks_clean  = []

    for i, mask in enumerate(tqdm(masks_raw, desc='Post-processing masks')):
        if i == 0:
            ref_frame    = filter_small_cells(mask.astype(np.int32), min_cell_size_px)
            origin_frame = ref_frame.copy()
            masks_clean.append(ref_frame)
        else:
            frame_merged = merge_to_reference(
                ref_frame, mask.astype(np.int32), origin_frame,
                iou_thresh=iou_thresh, ioq_thresh=ioq_thresh,
                dilation_px=dilation_px, verbose=verbose)
            frame_clean  = filter_small_cells(frame_merged, min_cell_size_px)
            masks_clean.append(frame_clean)
            ref_frame    = frame_clean

    return masks_clean, origin_frame


def interpolate_missing_masks(masks_clean, expected_ids, max_gap=4):
    """
    Fill short gaps (≤ max_gap frames) where a cell is absent by
    carrying forward the nearest present mask.
    """
    n      = len(masks_clean)
    result = [m.copy() for m in masks_clean]

    for cid in expected_ids:
        present = [t for t in range(n) if cid in np.unique(result[t])]
        if not present:
            print(f'  cell {cid}: never present, skipping')
            continue
        for idx in range(len(present) - 1):
            t_before, t_after = present[idx], present[idx + 1]
            gap = t_after - t_before - 1
            if gap == 0:
                continue
            if gap > max_gap:
                print(f'  cell {cid}: gap {t_before+1}–{t_after-1} '
                      f'({gap} frames) too large, skipping')
                continue
            print(f'  cell {cid}: interpolating frames {t_before+1}–{t_after-1}')
            mask_before = (result[t_before] == cid)
            mask_after  = (result[t_after]  == cid)
            for t_fill in range(t_before + 1, t_after):
                alpha        = (t_fill - t_before) / (t_after - t_before)
                interp_mask  = mask_before if alpha <= 0.5 else mask_after
                result[t_fill][interp_mask] = cid

    return result


# ══════════════════════════════════════════════════════════════════════════════
# 4. SPARK QUANTIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def quantify_spark(path, tracked_masks, read_frame_fn,
                   puncta_cell_percentile=99,
                   min_puncta_size=5,
                   max_puncta_size=1500,
                   min_puncta_mean_int=2600,
                   minutes_per_frame=4.0,
                   frame_offset_min=0.0):
    """
    Quantify SPARK signal for every cell in every frame.

    Parameters
    ----------
    path               : str  — path to source TIFF (for tifffile.imread per frame)
    tracked_masks      : (T, Y, X) int array
    read_frame_fn      : callable(t) → (Y, X) float32 — use make_read_frame()
    puncta_cell_percentile : percentile threshold within each cell
    min/max_puncta_size    : px² size filter for puncta objects
    min_puncta_mean_int    : mean intensity floor per punctum object
    minutes_per_frame      : frame interval in minutes
    frame_offset_min       : time (min) at frame 0

    Returns
    -------
    df_spark : pd.DataFrame
    """
    n_frames = tracked_masks.shape[0]
    records  = []

    for t in tqdm(range(n_frames), desc='Quantifying SPARK signal'):
        frame_img   = read_frame_fn(t)
        label_frame = tracked_masks[t]
        cell_ids    = np.unique(label_frame); cell_ids = cell_ids[cell_ids != 0]

        for cid in cell_ids:
            cell_mask   = (label_frame == cid)
            cell_px     = frame_img[cell_mask]
            if cell_px.size == 0:
                continue
            cell_thresh = np.percentile(cell_px, puncta_cell_percentile)
            cell_total  = float(cell_px.sum())
            cell_area   = int(cell_mask.sum())

            puncta_mask = (frame_img >= cell_thresh) & cell_mask
            puncta_mask = remove_small_objects(puncta_mask, min_size=min_puncta_size)
            p_labeled   = measure.label(puncta_mask)
            for r in measure.regionprops(p_labeled, intensity_image=frame_img):
                if r.area > max_puncta_size or r.mean_intensity < min_puncta_mean_int:
                    puncta_mask[p_labeled == r.label] = False
            p_labeled = measure.label(puncta_mask)

            puncta_total = float(frame_img[puncta_mask].sum())
            puncta_count = int(p_labeled.max())
            puncta_area  = int(puncta_mask.sum())
            spark_signal = puncta_total / cell_total if cell_total > 0 else np.nan

            records.append({
                'frame':             t,
                'time_min':          frame_offset_min + t * minutes_per_frame,
                'cell_id':           int(cid),
                'cell_area_px':      cell_area,
                'cell_total_int':    cell_total,
                'cell_thresh':       float(cell_thresh),
                'puncta_count':      puncta_count,
                'puncta_area_px':    puncta_area,
                'puncta_total_int':  puncta_total,
                'spark_signal':      spark_signal,
            })

    return pd.DataFrame(records)


def normalise_spark(df_spark, norm_frame=None, ctrl_cell_ids=None):
    """
    Normalise SPARK signal.

    norm_frame     : int or None — divide each cell's trace by its value at this frame
    ctrl_cell_ids  : list or None — divide all values by mean of control cells
    """
    df = df_spark.copy()

    if norm_frame is not None:
        ref = (df[df['frame'] == norm_frame]
               .set_index('cell_id')['spark_signal']
               .rename('ref_signal'))
        df = df.join(ref, on='cell_id')
        df['spark_norm'] = df['spark_signal'] / df['ref_signal']
        df.drop(columns='ref_signal', inplace=True)
        print(f'Normalised to frame {norm_frame}')
    else:
        df['spark_norm'] = df['spark_signal']

    if ctrl_cell_ids is not None:
        ctrl_mean = df[df['cell_id'].isin(ctrl_cell_ids)]['spark_norm'].mean()
        df['spark_norm_ctrl'] = df['spark_norm'] / ctrl_mean
        print(f'Control-normalised (mean={ctrl_mean:.4f})')
    else:
        df['spark_norm_ctrl'] = df['spark_norm']

    return df


# ══════════════════════════════════════════════════════════════════════════════
# 5. HDF5 STORAGE
# ══════════════════════════════════════════════════════════════════════════════

def save_to_hdf5(h5_path, run_key, path, df_spark, tracked_masks,
                 read_frame_fn, spark_params):
    """
    Save a complete SPARK run to HDF5 (append-safe).

    spark_params : dict of quantification parameters for metadata
    """
    run_metadata = {
        'tif_path':               path,
        'date_processed':         datetime.now().isoformat(),
        **spark_params,
    }

    n_frames = tracked_masks.shape[0]
    print(f'Saving run: {run_key}')

    with h5py.File(h5_path, 'a') as hf:
        if run_key in hf:
            del hf[run_key]
            print(f'  Overwriting existing run: {run_key}')

        grp = hf.create_group(run_key)
        grp.attrs['metadata'] = json.dumps(run_metadata)

        # SPARK dataframe
        df_grp = grp.create_group('spark_df')
        for col in df_spark.columns:
            df_grp.create_dataset(col, data=df_spark[col].values, compression='gzip')
        print(f'  spark_df: {len(df_spark)} rows × {len(df_spark.columns)} cols')

        # tracked masks
        grp.create_dataset('tracked_masks',
                           data=tracked_masks.astype(np.int16),
                           compression='gzip', compression_opts=4,
                           chunks=(1, tracked_masks.shape[1], tracked_masks.shape[2]))
        print(f'  tracked_masks: {tracked_masks.shape}')

        # puncta coords (sparse)
        pm_grp = grp.create_group('puncta_coords')
        for t in tqdm(range(n_frames), desc='  Saving puncta coords'):
            frame_img   = read_frame_fn(t)
            label_frame = tracked_masks[t]
            cell_ids    = np.unique(label_frame); cell_ids = cell_ids[cell_ids != 0]
            t_grp       = pm_grp.create_group(str(t))
            for cid in cell_ids:
                cell_mask   = (label_frame == cid)
                cell_px     = frame_img[cell_mask]
                cell_thresh = np.percentile(cell_px, spark_params.get('puncta_cell_percentile', 99))
                puncta_mask = (frame_img >= cell_thresh) & cell_mask
                puncta_mask = remove_small_objects(
                    puncta_mask, min_size=spark_params.get('min_puncta_size', 5))
                p_labeled   = measure.label(puncta_mask)
                for r in measure.regionprops(p_labeled, intensity_image=frame_img):
                    if (r.area > spark_params.get('max_puncta_size', 1500) or
                            r.mean_intensity < spark_params.get('min_puncta_mean_int', 2600)):
                        puncta_mask[p_labeled == r.label] = False
                rows_px, cols_px = np.where(puncta_mask)
                coords = np.stack([rows_px, cols_px], axis=1).astype(np.int16) \
                         if rows_px.size > 0 else np.zeros((0, 2), dtype=np.int16)
                t_grp.create_dataset(str(cid), data=coords, compression='gzip')

    size_mb = os.path.getsize(h5_path) / 1e6
    print(f'\nDone → {h5_path}  ({size_mb:.1f} MB)')


def load_spark_run(h5_path, run_key):
    """
    Load a saved SPARK run from HDF5.

    Returns
    -------
    meta          : dict
    df            : pd.DataFrame
    tracked_masks : np.ndarray  (T, Y, X) int16
    puncta_fn     : callable(t, cid) → bool mask (Y, X)
    """
    with h5py.File(h5_path, 'r') as hf:
        grp  = hf[run_key]
        meta = json.loads(grp.attrs['metadata'])
        df   = pd.DataFrame({k: grp['spark_df'][k][:] for k in grp['spark_df'].keys()})
        masks = grp['tracked_masks'][:]
        shape = masks.shape[1:]

        puncta_store = {}
        for t_str in grp['puncta_coords'].keys():
            t = int(t_str)
            puncta_store[t] = {}
            for cid_str in grp['puncta_coords'][t_str].keys():
                coords = grp['puncta_coords'][t_str][cid_str][:]
                puncta_store[t][int(cid_str)] = coords

    def puncta_fn(t, cid):
        mask = np.zeros(shape, dtype=bool)
        if t in puncta_store and cid in puncta_store[t]:
            coords = puncta_store[t][cid]
            if coords.size > 0:
                mask[coords[:, 0], coords[:, 1]] = True
        return mask

    return meta, df, masks, puncta_fn


# ══════════════════════════════════════════════════════════════════════════════
# 6. FIGURES
# ══════════════════════════════════════════════════════════════════════════════

def plot_spark_overlay(h5_path, run_keys, cell_ids_per_run=None,
                       save_path=None):
    """
    Overlay normalised SPARK traces from multiple runs / conditions.

    Parameters
    ----------
    run_keys         : list of str
    cell_ids_per_run : dict {run_key: [cell_ids]} or None (all cells)
    save_path        : output PNG path or None
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    cmap    = plt.cm.get_cmap('tab10')

    for i, rk in enumerate(run_keys):
        meta, df, _, _ = load_spark_run(h5_path, rk)
        if cell_ids_per_run and rk in cell_ids_per_run:
            df = df[df['cell_id'].isin(cell_ids_per_run[rk])]
        for j, (cid, sub) in enumerate(df.groupby('cell_id')):
            sub = sub.sort_values('frame')
            ax.plot(sub['time_min'], sub['spark_norm_ctrl'],
                    color=cmap((i * 10 + j) % 20), lw=1.5, alpha=0.8,
                    label=f'{rk}  cell {cid}')

    ax.set_xlabel('Time (min)', fontsize=10)
    ax.set_ylabel('Normalised SPARK signal', fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=7, frameon=False, bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved → {save_path}')
    plt.show()


def make_paper_figure(path, tracked_masks, df_spark, read_frame_fn,
                      figure_frames, figure_cells, cell_colors,
                      spark_params, crop_pad=60,
                      minutes_per_frame=4.0, frame_offset_min=0.0,
                      save_prefix=None):
    """
    Generate publication-quality sub-figure: cell rows × frame columns + trace panel.

    Parameters
    ----------
    path            : str  — source TIFF path
    tracked_masks   : (T, Y, X) int
    df_spark        : pd.DataFrame  with spark_norm_ctrl column
    read_frame_fn   : callable(t) → (Y,X) float32
    figure_frames   : list of int  — frame indices to display
    figure_cells    : list of int  — cell IDs to display (one row each)
    cell_colors     : dict {cell_id: hex_color}
    spark_params    : dict with puncta_cell_percentile / min_puncta_size etc.
    crop_pad        : px padding around union cell bbox
    save_prefix     : path prefix for .pdf/.png output, or None
    """
    pcp  = spark_params.get('puncta_cell_percentile', 99)
    mps  = spark_params.get('min_puncta_size', 5)
    maps = spark_params.get('max_puncta_size', 1500)
    mmi  = spark_params.get('min_puncta_mean_int', 2600)

    n_frames = len(figure_frames)
    H_img, W_img = tracked_masks.shape[1], tracked_masks.shape[2]

    # pre-compute union bbox per cell across all displayed frames
    crop_windows = {}
    for cid in figure_cells:
        r0_a, r1_a, c0_a, c1_a = np.inf, 0, np.inf, 0
        for t in figure_frames:
            cm = (tracked_masks[t] == cid)
            if cm.sum() == 0:
                continue
            rm, cm_ = np.where(cm)
            r0_a = min(r0_a, rm.min()); r1_a = max(r1_a, rm.max())
            c0_a = min(c0_a, cm_.min()); c1_a = max(c1_a, cm_.max())
        if np.isinf(r0_a):
            crop_windows[cid] = (0, 240, 0, 240)
        else:
            crop_windows[cid] = (
                max(int(r0_a) - crop_pad, 0), min(int(r1_a) + crop_pad, H_img),
                max(int(c0_a) - crop_pad, 0), min(int(c1_a) + crop_pad, W_img),
            )

    # figure sizing
    sample_h = max(r1 - r0 for r0, r1, c0, c1 in crop_windows.values())
    sample_w = max(c1 - c0 for r0, r1, c0, c1 in crop_windows.values())
    col_w    = 1.8
    fig_w    = col_w * (n_frames + 1.6)
    fig_h    = col_w * (sample_h / max(sample_w, 1)) * len(figure_cells) + 0.8

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=300)
    gs  = gridspec.GridSpec(
        len(figure_cells), n_frames + 1,
        width_ratios=[1] * n_frames + [1.6],
        wspace=0.04, hspace=0.12,
        left=0.04, right=0.97, top=0.88, bottom=0.14,
    )

    for row, cid in enumerate(figure_cells):
        r0, r1, c0, c1 = crop_windows[cid]

        for col, t in enumerate(figure_frames):
            ax = fig.add_subplot(gs[row, col])
            frame_img = read_frame_fn(t)
            cell_mask = (tracked_masks[t] == cid)

            if cell_mask.sum() == 0:
                ax.axis('off')
                if row == 0:
                    ax.set_title(f'{frame_offset_min + t * minutes_per_frame:.0f} min',
                                 fontsize=7.5, fontweight='bold', pad=3)
                continue

            # puncta
            cell_px     = frame_img[cell_mask]
            cell_thresh = np.percentile(cell_px, pcp)
            cell_total  = float(cell_px.sum())
            puncta_mask = (frame_img >= cell_thresh) & cell_mask
            puncta_mask = remove_small_objects(puncta_mask, min_size=mps)
            p_labeled   = measure.label(puncta_mask)
            for r in measure.regionprops(p_labeled, intensity_image=frame_img):
                if r.area > maps or r.mean_intensity < mmi:
                    puncta_mask[p_labeled == r.label] = False
            spark = float(frame_img[puncta_mask].sum()) / cell_total if cell_total > 0 else 0

            raw_crop    = frame_img[r0:r1, c0:c1].copy()
            cell_crop   = cell_mask[r0:r1,   c0:c1].copy()
            puncta_crop = puncta_mask[r0:r1,  c0:c1].copy()

            lo, hi = np.percentile(raw_crop[cell_crop], [1, 99.5])
            disp   = np.clip((raw_crop - lo) / (hi - lo + 1e-6), 0, 1)

            ax.imshow(disp, cmap='gray', interpolation='nearest', aspect='equal')

            boundary = find_boundaries(cell_crop, mode='outer')
            bd_rgba  = np.zeros((*boundary.shape, 4))
            bd_rgba[boundary, :3] = plt.matplotlib.colors.to_rgb(cell_colors[cid])
            bd_rgba[boundary,  3] = 0.9
            ax.imshow(bd_rgba, interpolation='nearest', aspect='equal')

            if puncta_crop.any():
                p_rgba = np.zeros((*puncta_crop.shape, 4))
                p_rgba[puncta_crop, 0] = 1.0
                p_rgba[puncta_crop, 1] = 1.0
                p_rgba[puncta_crop, 3] = 0.85
                ax.imshow(p_rgba, interpolation='nearest', aspect='equal')

            ax.axis('off')
            if row == 0:
                ax.set_title(f'{frame_offset_min + t * minutes_per_frame:.0f} min',
                             fontsize=7.5, fontweight='bold', pad=3, color='#222222')
            ax.text(0.04, 0.04, f'SPARK={spark:.3f}', transform=ax.transAxes,
                    fontsize=6.5, color='white', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.45, lw=0))
            if col == 0:
                ax.set_ylabel(f'Cell {cid}', fontsize=8, fontweight='bold',
                              color=cell_colors[cid], labelpad=4)
                ax.yaxis.set_label_position('left')
                ax.yaxis.label.set_visible(True)

    # trace panel
    ax_tr = fig.add_subplot(gs[:, n_frames])
    for cid in figure_cells:
        sub   = df_spark[df_spark['cell_id'] == cid].sort_values('frame')
        ax_tr.plot(sub['time_min'], sub['spark_norm_ctrl'],
                   color=cell_colors[cid], lw=1.5, label=f'Cell {cid}')
        shown = sub[sub['frame'].isin(figure_frames)]
        ax_tr.scatter(shown['time_min'], shown['spark_norm_ctrl'],
                      color=cell_colors[cid], s=18, zorder=5)
    for t in figure_frames:
        ax_tr.axvline(frame_offset_min + t * minutes_per_frame,
                      color='grey', lw=0.6, ls='--', alpha=0.5)
    ax_tr.set_xlabel('Time (min)', fontsize=8)
    ax_tr.set_ylabel('Normalised SPARK signal', fontsize=8)
    ax_tr.set_title('SPARK signal', fontsize=8, fontweight='bold')
    ax_tr.tick_params(labelsize=7)
    ax_tr.spines[['top', 'right']].set_visible(False)
    ax_tr.legend(fontsize=7, frameon=False)

    legend_els = [mpatches.Patch(facecolor=cell_colors[c], alpha=0.8,
                                  label=f'Cell {c} boundary') for c in figure_cells]
    legend_els += [mpatches.Patch(facecolor='yellow', alpha=0.8, label='SPARK puncta')]
    fig.legend(handles=legend_els, loc='lower center', ncol=3,
               fontsize=7, frameon=False, bbox_to_anchor=(0.42, 0.01))

    plt.tight_layout()
    if save_prefix:
        plt.savefig(f'{save_prefix}.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'{save_prefix}.png', dpi=300, bbox_inches='tight')
        print(f'Saved → {save_prefix}.pdf / .png')
    plt.show()
    return fig
