import re
from typing import Dict, Optional


def _num(s: str) -> Optional[float]:
    """Parse a number that may contain commas and unit suffixes.

    Returns a float or None if parsing fails.
    """
    if s is None:
        return None
    s = s.strip()
    # remove commas
    s = s.replace(',', '')
    # strip common units (uJ, mm^2, %) and parentheses
    s = re.sub(r"\s*(uJ|mJ|J|mm\^2|mm2|%|GHz)\s*$", '', s, flags=re.IGNORECASE)
    try:
        return float(s)
    except Exception:
        return None


def parse_timeloop_stats(path: str) -> Dict[str, Optional[float]]:
    """Parse a timeloop-mapper.stats.txt file and extract useful metrics.

    Extracted fields (when present):
      - gflops (GFLOPs @1GHz)
      - utilization_pct
      - cycles
      - energy_uJ
      - edp
      - area_mm2
      - total_ops
      - total_memory_accesses
      - optimal_ops_per_byte
      - algorithmic_intensity_ops_per_access = total_ops / total_memory_accesses
      - algorithmic_intensity_ops_per_byte = optimal_ops_per_byte if present, else same as ops_per_access

    Returns a dict mapping keys to floats or None.
    """
    metrics = {
        'gflops': None,
        'utilization_pct': None,
        'cycles': None,
        'energy_uJ': None,
        'edp': None,
        'area_mm2': None,
        'total_ops': None,
        'total_memory_accesses': None,
        'optimal_ops_per_byte': None,
        'algorithmic_intensity_ops_per_access': None,
        'algorithmic_intensity_ops_per_byte': None,
    }

    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()

    # Summary Stats block
    m = re.search(r'GFLOPs\s*\(@1GHz\):\s*([0-9.,Ee+-]+)', text)
    if m:
        metrics['gflops'] = _num(m.group(1))

    m = re.search(r'Utilization:\s*([0-9.,]+)%', text)
    if m:
        metrics['utilization_pct'] = _num(m.group(1))

    m = re.search(r'Cycles:\s*([0-9,]+)', text)
    if m:
        metrics['cycles'] = _num(m.group(1))

    m = re.search(r'Energy:\s*([0-9.,Ee+-]+)\s*uJ', text, flags=re.IGNORECASE)
    if m:
        metrics['energy_uJ'] = _num(m.group(1))

    m = re.search(r'EDP\(J\*cycle\):\s*([0-9.,Ee+-]+)', text)
    if m:
        metrics['edp'] = _num(m.group(1))

    m = re.search(r'Area:\s*([0-9.,Ee+-]+)\s*mm', text)
    if m:
        metrics['area_mm2'] = _num(m.group(1))
    else:
        # try matching "Area: 0.00 mm^2" or similar
        m = re.search(r'Area:\s*([0-9.,Ee+-]+)\s*mm\^2', text)
        if m:
            metrics['area_mm2'] = _num(m.group(1))

    # Operational Intensity & totals
    m = re.search(r'Total ops\s*:\s*([0-9,]+)', text)
    if m:
        metrics['total_ops'] = _num(m.group(1))

    m = re.search(r'Total memory accesses required\s*:\s*([0-9,]+)', text)
    if m:
        metrics['total_memory_accesses'] = _num(m.group(1))

    m = re.search(r'Optimal Op per Byte\s*:\s*([0-9.,Ee+-]+)', text)
    if m:
        metrics['optimal_ops_per_byte'] = _num(m.group(1))

    # Fallbacks: sometimes words are slightly different
    if metrics['total_ops'] is None:
        m = re.search(r'Total elementwise ops\s*:\s*([0-9,]+)', text)
        if m:
            total_elem = _num(m.group(1))
        else:
            total_elem = None
        m = re.search(r'Total reduction ops\s*:\s*([0-9,]+)', text)
        if m:
            total_red = _num(m.group(1))
        else:
            total_red = None
        if total_elem is not None and total_red is not None:
            metrics['total_ops'] = total_elem + total_red

    if metrics['total_memory_accesses'] is None:
        m = re.search(r'Total memory accesses required\s*:\s*([0-9,]+)', text)
        if m:
            metrics['total_memory_accesses'] = _num(m.group(1))

    # Compute algorithmic intensity
    if metrics['total_ops'] and metrics['total_memory_accesses']:
        try:
            metrics['algorithmic_intensity_ops_per_access'] = (
                float(metrics['total_ops']) / float(metrics['total_memory_accesses'])
            )
        except Exception:
            metrics['algorithmic_intensity_ops_per_access'] = None

    # If optimal op/byte present, prefer that as ops/byte metric
    if metrics['optimal_ops_per_byte'] is not None:
        metrics['algorithmic_intensity_ops_per_byte'] = metrics['optimal_ops_per_byte']
    elif metrics['algorithmic_intensity_ops_per_access'] is not None:
        # We don't strictly know whether "memory accesses" is in bytes or words.
        # Use ops_per_access as a fallback for ops/byte.
        metrics['algorithmic_intensity_ops_per_byte'] = metrics['algorithmic_intensity_ops_per_access']

    return metrics


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python parse_timeloop_stats.py /path/to/timeloop-mapper.stats.txt')
        raise SystemExit(2)
    path = sys.argv[1]
    out = parse_timeloop_stats(path)
    # Print a compact report
    print('Parsed Timeloop stats:')
    for k, v in out.items():
        print(f'  {k}: {v}')
