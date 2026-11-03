"""
Per-instance context: the artifacts of the Certify stage, computed lazily, timed and cached.

Every artifact records its own computing time. A method that uses an artifact is charged that time
(once per run), so that budget comparisons are fair: a method using the relaxation pays for it.

Artifacts
  lp            (P_LP, alpha*)                      LP vertex (primal simplex)
  attainable    sorted attainable loads (units)
  lb_cut        (LB_cut, alpha_cut, Q)              Theorem 4.19 (bisection over attainable loads)
  lb_conf       (LB_conf, alpha_conf, finished)      Theorem 4.20 (column generation, time limit)
  cert_lp / cert_cut / cert_conf                     Certificates objects (exact slot laws)
  bounds        dict of every lower and upper certificate (Theorems 3.1-4.23)
  ir            (x, peak, guarantee, rule_ok)       Theorem 4.21
"""

import time

import numpy as np

from . import certificates as C
from . import laws as B
from .instance import UNIT


class Context:
    def __init__(self, inst, env=None, cfg=None):
        self.inst, self.env = inst, env
        self.cfg = cfg or {}
        self._val, self._own = {}, {}
        self.depends = {k: tuple(v) for k, v in DEPENDS.items()}
        if not self.cfg.get("use_conf", True):
            self.depends["lb"] = ("cert_lp", "lb_cut")

    # ---- generic lazy evaluation ----
    def get(self, key):
        """Value of an artifact; computed on first use. Each artifact times only its own work
        (its dependencies are obtained through get() before the timer starts)."""
        if key not in self._val:
            self._val[key] = getattr(self, "_make_" + key)()
        return self._val[key]

    def has(self, key):
        return key in self._val

    def charge(self, keys):
        """Total computing time of the given artifacts and of everything they depend on."""
        seen, stack, tot = set(), list(keys), 0.0
        while stack:
            k = stack.pop()
            if k in seen:
                continue
            seen.add(k)
            self.get(k)
            tot += self.own_cost.get(k, 0.0)
            stack.extend(self.depends.get(k, ()))
        return tot

    def closure(self, keys):
        """The artifacts needed to obtain the given ones (themselves and their dependencies)."""
        seen, stack = set(), list(keys)
        while stack:
            k = stack.pop()
            if k in seen:
                continue
            seen.add(k)
            stack.extend(self.depends.get(k, ()))
        return seen

    def cost_of(self, keys, exclude=()):
        """Recorded computing time of the given artifacts and their dependencies, except those in exclude
        (computed if needed; the computation itself is not timed by the caller)."""
        new = self.closure(keys) - set(exclude)
        for k in new:
            self.get(k)
        return sum(self.own_cost.get(k, 0.0) for k in new), new

    @property
    def own_cost(self):
        return self._own

    def _timed(self, key, f):
        t0 = time.time()
        v = f()
        self._own[key] = time.time() - t0
        return v

    def __getstate__(self):
        d = self.__dict__.copy(); d["env"] = None
        return d

    # ---- artifacts ----
    def _make_lp(self):
        return self._timed("lp", lambda: B.solve_lp(self.inst, self.env))

    def _make_cert_lp(self):
        p_lp, a = self.get("lp")
        return self._timed("cert_lp", lambda: C.Certificates(self.inst, a, p_lp))

    def _make_attainable(self):
        cert = self.get("cert_lp")
        return self._timed("attainable", lambda: cert.attainable_loads())

    def _ub_level(self):
        p_lp, _ = self.get("lp")
        return self.get("cert_lp").round_down_ub(p_lp + self.inst.w_max)

    def _make_lb_cut(self):
        cert = self.get("cert_lp"); self.get("attainable")
        kmax = self.cfg.get("cut_kmax", 8)
        return self._timed("lb_cut", lambda: C.cut_lower_bound(cert, self._ub_level(), self.env, kmax))

    def _make_lb_conf(self):
        cert = self.get("cert_lp"); lbc = self.get("lb_cut")[0]
        if self.inst.n > self.cfg.get("conf_max_n", 300):
            self._own["lb_conf"] = 0.0
            return (lbc, None, False)
        tl = self.cfg.get("conf_time_limit", 60.0)
        return self._timed("lb_conf", lambda: _conf_with_limit(cert, lbc, self._ub_level(), self.env, tl))

    def _make_cert_cut(self):
        lbc, ac, q = self.get("lb_cut")
        return self._timed("cert_cut", lambda: C.Certificates(self.inst, ac, q))

    def _make_cert_conf(self):
        lbf, af, ok = self.get("lb_conf")
        if af is None:
            self._own["cert_conf"] = 0.0
            return None
        return self._timed("cert_conf", lambda: C.Certificates(self.inst, af, lbf))

    def _make_lb(self):
        """Certified lower bound (8)/(29): max of the rounded LP, LB_cut and LB_conf (if computed)."""
        p_lp, _ = self.get("lp"); cert = self.get("cert_lp")
        lbc = self.get("lb_cut")[0]
        lbf = self.get("lb_conf")[0] if self.cfg.get("use_conf", True) else lbc
        self._own["lb"] = 0.0
        return max(cert.round_up_lb(p_lp), lbc, lbf)

    def _make_ir(self):
        p_lp, _ = self.get("lp")
        if self.inst.n > self.cfg.get("ir_max_n", 2000):
            self._own["ir"] = 0.0
            return None
        return self._timed("ir", lambda: C.iterative_rounding(self.inst, p_lp, self.env))

    def _make_bounds(self):
        for d in self.depends["bounds"]:
            self.get(d)
        return self._timed("bounds", self._bounds)

    def _bounds(self):
        inst = self.inst
        p_lp, a = self.get("lp")
        cert = self.get("cert_lp")
        lbc, ac, q = self.get("lb_cut")
        lbf, af, conf_ok = self.get("lb_conf")
        out = {"P_LP": p_lp, "LB_LP_rounded": cert.round_up_lb(p_lp), "LB_cut": lbc, "LB_conf": lbf,
               "conf_finished": bool(conf_ok), "w_max": inst.w_max, "LP_plus_wmax": p_lp + inst.w_max}
        mats = {"lp": cert, "cut": self.get("cert_cut")}
        if self.get("cert_conf") is not None:
            mats["conf"] = self.get("cert_conf")
        old = B.old_bounds(inst, a)
        out.update({f"published_{k}": v for k, v in old.items()})
        lim = self.cfg.get("exact_dp_max_n", 400)
        for tag, c in mats.items():
            t0 = time.time()
            for k in ("union", "component_union", "lll", "averaging", "component_averaging"):
                out[f"{tag}_{k}"] = getattr(c, k)()
            for k, v in c.closed_forms().items():
                out[f"{tag}_closed_{k}"] = v
            drc = C.DRCertificates(inst, c.alpha, c.p_lp)
            for k, v in drc.all().items():
                out[f"{tag}_{k}"] = v
            if inst.n <= lim:
                for mode, name in (("systematic", "exist_sys"), ("compatible", "exist_full")):
                    try:
                        f = C.ForestDP(c, mode)
                        out[f"{tag}_{name}"] = f.threshold()
                        if mode == "systematic" and tag == "lp":
                            out["lp_emax_exact"] = f.expected_max()
                    except (RuntimeError, MemoryError):
                        out[f"{tag}_{name}"] = np.nan
            out[f"time_certificates_{tag}"] = time.time() - t0
        ir = self.get("ir")
        out["IR_guarantee"] = np.floor(ir[2] * UNIT + 1e-4) / UNIT if ir is not None else np.nan
        uppers = [v for k, v in out.items() if _is_upper(k) and v == v]
        out["U_best"] = cert.round_down_ub(min(uppers))
        out["LB"] = max(out["LB_LP_rounded"], lbc, lbf)
        out["a_priori_optimal"] = abs(out["U_best"] - out["LB"]) < 1e-9
        return out


DEPENDS = {"cert_lp": ("lp",), "attainable": ("cert_lp",), "lb_cut": ("cert_lp", "attainable"),
           "lb_conf": ("lb_cut",), "cert_cut": ("lb_cut",), "cert_conf": ("lb_conf",), "ir": ("lp",),
           "lb": ("cert_lp", "lb_cut", "lb_conf"),
           "bounds": ("cert_lp", "lb_cut", "lb_conf", "cert_cut", "cert_conf", "ir")}

LOWER_KEYS = ("P_LP", "LB_LP_rounded", "LB_cut", "LB_conf", "LB")


def _is_upper(k):
    if k.startswith("time_") or k in LOWER_KEYS or k in ("w_max", "conf_finished", "a_priori_optimal", "U_best"):
        return False
    if k.startswith("published_d_all") or k.endswith("closed_d_random") or k.endswith("closed_largest_component") \
            or k.endswith("closed_n_components"):
        return False
    return True


def _conf_with_limit(cert, lb_cut, ub, env, time_limit):
    """Configuration bound by bisection with a time limit. On timeout returns the smallest attainable
    load above the largest level proved infeasible (or LB_cut), and no matrix."""
    inst, A = cert.inst, cert.attainable_loads()
    lo = int(np.searchsorted(A, int(np.ceil(lb_cut * UNIT - 1e-6))))
    hi = int(np.searchsorted(A, int(np.floor(ub * UNIT + 1e-6)), side="right")) - 1
    deadline = time.time() + time_limit
    pool, best_alpha = [], None
    while lo < hi:
        mid = (lo + hi) // 2
        ok, al, pool = C.config_lp(inst, int(A[mid]), env, pool, deadline=deadline)
        if ok is None:                                   # time limit reached inside the column generation
            return (A[lo] / UNIT, None, False)
        if ok:
            hi, best_alpha = mid, al
        else:
            lo = mid + 1
    if best_alpha is None:
        ok, best_alpha, pool = C.config_lp(inst, int(A[lo]), env, pool, deadline=deadline)
        if ok is None:
            return (A[lo] / UNIT, None, False)
    return (A[lo] / UNIT, best_alpha, True)


def make_context(inst, env, cfg):
    return Context(inst, env, cfg)
