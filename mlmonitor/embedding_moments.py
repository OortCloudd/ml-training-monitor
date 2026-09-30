"""Scalar representation diagnostics from explicitly mapped sample moments.

Inputs are sample standard deviations (ddof=1), not model weights or images.
No diagnostic threshold or training action is defined here.
"""
import math

from .storage import finite


DERIVED_KEYS = ('total_variance', 'centered_feature_rms', 'centered_energy_fraction')
SOURCE_KEYS = ('per_dimension_std', 'n_samples', 'n_dimensions', 'norm_mean', 'norm_std')


def derive(values):
    """Return finite scalars or null; never invent a missing moment or epsilon."""
    result = dict.fromkeys(DERIVED_KEYS)
    count, dimensions = values.get('n_samples'), values.get('n_dimensions')
    stds = values.get('per_dimension_std')
    if (type(count) is not int or count < 2 or type(dimensions) is not int or dimensions < 1
            or not isinstance(stds, list) or len(stds) != dimensions):
        return result
    try:
        if any(finite(value) is None or value < 0 for value in stds):
            return result
        variance = math.fsum(value * value for value in stds)
        if not math.isfinite(variance):
            return result
        result['total_variance'] = variance
        result['centered_feature_rms'] = math.sqrt(variance / dimensions)
        norm_mean, norm_std = finite(values.get('norm_mean')), finite(values.get('norm_std'))
        if norm_mean is None or norm_mean < 0 or norm_std is None or norm_std < 0:
            return result
        correction = (count - 1) / count
        total_energy = norm_mean * norm_mean + correction * norm_std * norm_std
        if not math.isfinite(total_energy) or total_energy <= 0:
            return result
        fraction = correction * variance / total_energy
        if math.isfinite(fraction):
            result['centered_energy_fraction'] = fraction
    except (OverflowError, ValueError):
        pass
    return result
