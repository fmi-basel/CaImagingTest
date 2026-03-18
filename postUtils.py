import re
import numpy as np


def normalize_odor_name(value):
    if value is None:
        return None
    return str(value).strip().upper()


def safe_filename(value):
    return re.sub(r'[^A-Za-z0-9._-]+', '_', str(value))


def pvalue_to_stars(p_value):
    if not np.isfinite(p_value):
        return 'n/a'
    if p_value < 0.001:
        return '***'
    if p_value < 0.01:
        return '**'
    if p_value < 0.05:
        return '*'
    return 'ns'


def sem(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size <= 1:
        return np.nan
    return float(np.std(values, ddof=1) / np.sqrt(values.size))


def nanmean_padded(traces):
    valid_traces = [np.asarray(trace, dtype=float) for trace in traces if np.asarray(trace).size > 0]
    if len(valid_traces) == 0:
        return np.array([], dtype=float)

    max_len = max(trace.size for trace in valid_traces)
    padded = np.full((len(valid_traces), max_len), np.nan, dtype=float)
    for trace_idx, trace in enumerate(valid_traces):
        padded[trace_idx, :trace.size] = trace
    return np.nanmean(padded, axis=0)


def mean_and_sem_padded(traces):
    valid_traces = [np.asarray(trace, dtype=float) for trace in traces if np.asarray(trace).size > 0]
    if len(valid_traces) == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    max_len = max(trace.size for trace in valid_traces)
    padded = np.full((len(valid_traces), max_len), np.nan, dtype=float)
    for trace_idx, trace in enumerate(valid_traces):
        padded[trace_idx, :trace.size] = trace

    mean_trace = np.nanmean(padded, axis=0)
    n_non_nan = np.sum(np.isfinite(padded), axis=0)
    sem_trace = np.full(mean_trace.shape, np.nan, dtype=float)
    valid_n = n_non_nan > 1
    sem_trace[valid_n] = np.nanstd(padded[:, valid_n], axis=0, ddof=1) / np.sqrt(n_non_nan[valid_n])
    return mean_trace, sem_trace
