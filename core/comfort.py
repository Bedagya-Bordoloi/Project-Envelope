import math
from pythermalcomfort.models import pmv_ppd_iso

class PMVOutOfRangeError(Exception):
    pass

def seasonal_clo(t_out):
    """Linear interpolation: 1.2 clo at 0C -> 0.5 clo at 26C."""
    if t_out <= 0: return 1.2
    if t_out >= 26: return 0.5
    return 1.2 - (0.7 * (t_out / 26.0))

def calculate_pmv(temp, humidity, t_out):
    clo = seasonal_clo(t_out)
    result = pmv_ppd_iso(
        tdb=temp, tr=temp, vr=0.1, rh=humidity,
        met=1.2, clo=clo, model="7730-2005"
    )
    pmv_val = float(result.pmv)
    if math.isnan(pmv_val):
        raise PMVOutOfRangeError("PMV Out of Range")
    return pmv_val, clo