"""Comparing configurations that were all measured on the same folds/drivers.

THE ONE implementation of the blocked (paired) comparison. Three selection
routines needed it — the ANIL sweep, the population sweep and the tau sweep —
and each had independently written the *unblocked* version, so this module
exists for the same reason ``head_adapt`` does: the alternative is three
estimators that are supposed to agree and quietly do not.

THE FAILURE IT FIXES
--------------------
Every sweep here is fully crossed: each configuration is run on each fold (or
scored on each driver), so the design is balanced. Balance makes the *mean*
comparison valid — the fold offsets cancel — and that is exactly what makes the
bug survive review. Write one run's score as

    score(config c, block b) = mu + alpha_c + beta_b + eps

and note that ``sd`` taken over one config's runs sees ``beta_b + eps`` only:
``alpha_c`` is constant within a config and contributes nothing. On this cohort
``beta`` (which driver / which fold) is an order of magnitude larger than
``alpha`` (which config) — between-driver sd is 0.343 against a tau grid
spanning 0.028 — so ``se = sd/sqrt(n)`` is essentially a measurement of driver
difficulty wearing the label of a configuration error bar.

That matters because every one of these routines then does:

    best   = min(table, key=mean)          # correct, unaffected by blocks
    within = [t for t in table if t.mean <= best.mean + best.se]
    pick   = max(within, key=<more regularization>)

``best`` is computed and discarded; ``pick`` is what ships. With an inflated
``se`` the band swallows the whole table, the tie-break fires unconditionally,
and the routine returns the most-regularized corner of the grid having used no
information from the data at all. The tie-break rule itself is sound — the ties
were fabricated by a mis-specified standard error.

Because both configs are measured on the SAME blocks, the block term cancels in
their difference:

    score(A,b) - score(B,b) = alpha_A - alpha_B + (eps_A - eps_B)

so the relevant spread is ``eps``, not ``beta``. This is the paired vs unpaired
t-test distinction, and the reason you would never analyse a within-subject
experiment with an unpaired test.

**Balance fixes the point estimate, not the error bar.**
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

__all__ = ["blocked_effects"]


def blocked_effects(cells: Iterable[Tuple[Any, Any, float]],
                    ) -> Optional[Dict[Any, Tuple[float, float, int]]]:
    """Config effects with the block removed. ``{config: (mean, se, n_blocks)}``.

    ``cells`` is an iterable of ``(config_key, block_key, value)``. Repeats of the
    same (config, block) are averaged first, so a config run at several seeds
    inside one fold contributes one number rather than n.

    The returned mean is the blocked estimate with the grand mean added back, so
    it stays on the scale of the original metric (a set-MAE looks like a set-MAE)
    and is directly comparable to the unblocked mean it replaces — in a balanced
    design the two are equal, and the whole difference is in ``se``.

    Only blocks where EVERY config is present are used. A block missing one
    config would otherwise shift that block's centering constant and hand the
    configs that did run there a spurious advantage. ``n_blocks`` is returned so
    the caller can report how much of the design survived that filter.

    Returns ``None`` when fewer than two complete blocks remain — there is then
    no blocked estimate to make, and the caller must fall back to the unblocked
    mean AND SAY SO, rather than silently reporting a number of unknown type.
    """
    acc: Dict[Any, Dict[Any, List[float]]] = {}
    for cfg, blk, val in cells:
        v = float(val)
        if v != v:                      # NaN: a failed run is absent, not zero
            continue
        acc.setdefault(cfg, {}).setdefault(blk, []).append(v)
    if len(acc) < 2:
        return None

    configs = list(acc)
    complete = sorted(
        set.intersection(*(set(acc[c]) for c in configs)),
        key=repr,
    )
    if len(complete) < 2:
        return None

    cell = {c: {b: float(np.mean(acc[c][b])) for b in complete} for c in configs}
    grand = float(np.mean([cell[c][b] for c in configs for b in complete]))
    out: Dict[Any, Tuple[float, float, int]] = {}
    for c in configs:
        resid = np.array([cell[c][b] - np.mean([cell[c2][b] for c2 in configs])
                          for b in complete])
        se = (float(resid.std(ddof=1) / np.sqrt(resid.size))
              if resid.size > 1 else float("nan"))
        out[c] = (float(resid.mean()) + grand, se, len(complete))
    return out
