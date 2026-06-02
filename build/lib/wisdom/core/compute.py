# wisdom/core/compute.py
from typing import Dict, List, Tuple

def combinations_coverage(assignments: List[Dict[str, int]],
                          cluster_sizes: Dict[str, int]) -> Tuple[float, float, int]:
    keys = sorted(cluster_sizes.keys())
    total = 1
    for k in keys: total *= int(cluster_sizes[k])
    seen = set()
    for a in assignments:
        tup = tuple(a.get(k, -1) for k in keys)
        if -1 not in tup: seen.add(tup)
    rate = len(seen)/total if total>0 else 0.0
    max_cov = min(1.0, len(assignments)/total if total>0 else 0.0)
    return rate, total, max_cov