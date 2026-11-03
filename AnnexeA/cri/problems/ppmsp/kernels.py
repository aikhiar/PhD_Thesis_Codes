"""
Numba kernels of all combinatorial components (schedules x are int8 n x T, loads are int64 units).

Earlier methods, faithful to the logic of the original code (Main Notebook.ipynb), with
incremental evaluation instead of full recomputation:
  greedy_vf          greedy valley filling (G)
  rr_chunk           randomized resampling (RR), N iterations split in chunks
  rlp_trial          randomized LP rounding with peak-aware repair (RLP), one trial
  sa_chunk           simulated annealing (SA) with partial slot reassignment (eps = 0: original)
New methods:
  ce_run             derandomized rounding by conditional expectations (CE): excess / union
                     estimator, optional restriction to subsets compatible with alpha (componentwise)
  mt_run             Moser-Tardos resampling with systematic sampling (MT)
  lls_iterate        load-levelling iterated local search (LLS): descent, ejection chains, kicks
Long-running kernels work in chunks so that the Python side can enforce wall-clock budgets and
record anytime traces; they return the iteration numbers at which the best value improved.
"""

import numpy as np
from numba import njit

BIG = np.int64(1) << 60


# ── basics ──────────────────────────────────────────────────────────────────────

@njit(cache=True)
def seed_numba(seed):
    np.random.seed(seed)


@njit(cache=True)
def loads_of(x, wu):
    n, T = x.shape
    L = np.zeros(T, np.int64)
    for i in range(n):
        wi = wu[i]
        for t in range(T):
            if x[i, t]:
                L[t] += wi
    return L


@njit(cache=True)
def sum_sq(L):
    s = 0.0
    for t in range(L.shape[0]):
        s += float(L[t]) * float(L[t])
    return s


@njit(cache=True)
def count_peak(L):
    pk = L.max()
    c = 0
    for t in range(L.shape[0]):
        if L[t] == pk:
            c += 1
    return pk, c


@njit(cache=True)
def seg_build(L):
    T = L.shape[0]
    size = 1
    while size < T:
        size *= 2
    tree = np.full(2 * size, -BIG, np.int64)
    for t in range(T):
        tree[size + t] = L[t]
    for k in range(size - 1, 0, -1):
        tree[k] = max(tree[2 * k], tree[2 * k + 1])
    return tree, size


@njit(cache=True)
def seg_set(tree, size, t, val):
    k = size + t
    tree[k] = val
    k //= 2
    while k >= 1:
        a = tree[2 * k]; b = tree[2 * k + 1]
        tree[k] = a if a > b else b
        k //= 2


@njit(cache=True)
def random_schedule(s, e, p, T):
    """Each vehicle on p_i distinct slots drawn uniformly in its window (original generate_schedule)."""
    n = s.shape[0]
    x = np.zeros((n, T), np.int8)
    for i in range(n):
        m = e[i] - s[i] + 1
        perm = np.random.permutation(m)
        for k in range(p[i]):
            x[i, s[i] + perm[k]] = 1
    return x


# ── G: greedy valley filling (original greedy_constructive_heuristic) ──────────

@njit(cache=True)
def greedy_vf(s, e, p, wu, T, seed):
    np.random.seed(seed)
    n = s.shape[0]
    x = np.zeros((n, T), np.int8)
    L = np.zeros(T, np.int64)
    order = np.argsort(-wu, kind="mergesort")
    for oi in range(n):
        i = order[oi]
        for r in range(p[i]):
            minv = BIG; cnt = 0
            for t in range(s[i], e[i] + 1):
                if x[i, t] == 0:
                    if L[t] < minv:
                        minv = L[t]; cnt = 1
                    elif L[t] == minv:
                        cnt += 1
            k = np.random.randint(cnt)
            for t in range(s[i], e[i] + 1):
                if x[i, t] == 0 and L[t] == minv:
                    if k == 0:
                        x[i, t] = 1; L[t] += wu[i]
                        break
                    k -= 1
    return x, L


# ── RR: randomized resampling (original moser_tardos_inspired_algorithm, P_max = 0) ──

@njit(cache=True)
def rr_chunk(x, L, s, e, p, wu, n_iter, best_x, best_val, rec_it, rec_val, it0):
    """n_iter iterations of RR on state (x, L). best_x / best_val are updated in place (best_val is a
    1-array). Improvements are written to rec_it / rec_val; returns the number written."""
    n, T = x.shape
    nrec = 0
    cand = np.empty(T, np.int64)
    veh = np.empty(n, np.int64)
    for k in range(n_iter):
        pk = L.max()
        if pk < best_val[0]:
            best_val[0] = pk
            best_x[:, :] = x
            if nrec < rec_it.shape[0]:
                rec_it[nrec] = it0 + k; rec_val[nrec] = pk; nrec += 1
        # violated slots: load >= current peak (original: power_per_slot >= P_threshold)
        nc = 0
        for t in range(T):
            if L[t] >= pk:
                cand[nc] = t; nc += 1
        ts = cand[np.random.randint(nc)]
        nv = 0
        for i in range(n):
            if x[i, ts]:
                veh[nv] = i; nv += 1
        if nv == 0:
            continue
        i = veh[np.random.randint(nv)]
        for t in range(s[i], e[i] + 1):
            if x[i, t]:
                x[i, t] = 0; L[t] -= wu[i]
        m = e[i] - s[i] + 1
        resid = np.empty(m, np.int64)
        for k2 in range(m):
            resid[k2] = L[s[i] + k2]
        idx = np.argsort(resid)
        for k2 in range(p[i]):
            t = s[i] + idx[k2]
            x[i, t] = 1; L[t] += wu[i]
    pk = L.max()
    if pk < best_val[0]:
        best_val[0] = pk
        best_x[:, :] = x
        if nrec < rec_it.shape[0]:
            rec_it[nrec] = it0 + n_iter; rec_val[nrec] = pk; nrec += 1
    return nrec


# ── RLP: randomized LP rounding with peak-aware repair (original lp_based_rounding_algorithm) ──

@njit(cache=True)
def _weighted_sample(slots, weights, k):
    """Sequential weighted sampling without replacement (original weighted_sample_slots)."""
    m = slots.shape[0]
    k = min(k, m)
    out = np.empty(k, np.int64)
    avail = np.ones(m, np.bool_)
    for r in range(k):
        tot = 0.0
        for j in range(m):
            if avail[j]:
                tot += weights[j]
        u = np.random.random() * tot
        acc = 0.0
        pick = -1
        for j in range(m):
            if avail[j]:
                acc += weights[j]
                pick = j
                if u <= acc:
                    break
        out[r] = slots[pick]; avail[pick] = False
    return out


@njit(cache=True)
def rlp_trial(alpha, s, e, p, wu, unit):
    """One rounding + repair trial. alpha: optimal LP solution (n x T). Returns (x, L)."""
    n, T = alpha.shape
    x = np.zeros((n, T), np.int8)
    for i in range(n):
        for t in range(s[i], e[i] + 1):
            if np.random.random() < alpha[i, t]:
                x[i, t] = 1
    L = loads_of(x, wu)
    for i in range(n):
        r = 0
        for t in range(s[i], e[i] + 1):
            r += x[i, t]
        wi_kw = wu[i] / unit
        if r < p[i]:
            m = 0
            for t in range(s[i], e[i] + 1):
                if x[i, t] == 0:
                    m += 1
            sl = np.empty(m, np.int64); wt = np.empty(m)
            m = 0
            for t in range(s[i], e[i] + 1):
                if x[i, t] == 0:
                    sl[m] = t; wt[m] = 1.0 / (L[t] / unit + wi_kw + 1e-6); m += 1
            add = _weighted_sample(sl, wt, p[i] - r)
            for t in add:
                x[i, t] = 1; L[t] += wu[i]
        elif r > p[i]:
            m = 0
            for t in range(s[i], e[i] + 1):
                if x[i, t] == 1:
                    m += 1
            sl = np.empty(m, np.int64); wt = np.empty(m)
            m = 0
            for t in range(s[i], e[i] + 1):
                if x[i, t] == 1:
                    sl[m] = t; wt[m] = L[t] / unit + 1e-6; m += 1
            rem = _weighted_sample(sl, wt, r - p[i])
            for t in rem:
                x[i, t] = 0; L[t] -= wu[i]
    return x, L


# ── SA: simulated annealing with partial slot reassignment (original simulated_annealing) ──

@njit(cache=True)
def sa_chunk(x, L, tree, size, s, e, p, wu, unit, state, params, best_x, best_L, rec_it, rec_val, lb_units):
    """Run SA until n_steps (params[6]) iterations are done, theta <= theta_min, the global iteration
    limit is reached or the best peak meets lb_units.
    state = [theta, f_cur, k, epoch_pos, best_peak, best_sq] (float64, updated in place)
    params = [theta_min, alpha_c, epoch_len, N, alpha_p, eps, n_steps]
    f = peak_kW + eps * sum_t (L_t in kW)^2; eps = 0 reproduces the original objective."""
    n, T = x.shape
    theta = state[0]; f_cur = state[1]; k = int(state[2]); ep = int(state[3])
    best_pk = state[4]; best_sq = state[5]
    theta_min = params[0]; alpha_c = params[1]; ell = int(params[2]); N = int(params[3])
    alpha_p = params[4]; eps = params[5]; n_steps = int(params[6])
    sq = sum_sq(L) / (unit * unit)
    nrec = 0
    rem = np.empty(T, np.int64); add = np.empty(T, np.int64)
    done = 0
    while theta > theta_min and k < N and done < n_steps and best_pk > lb_units:
        i = np.random.randint(n)
        kk = max(1, int(np.ceil(alpha_p * p[i])))
        # assigned slots
        na = 0
        for t in range(s[i], e[i] + 1):
            if x[i, t]:
                rem[na] = t; na += 1
        if na < kk:
            kk = na
        # choose kk distinct assigned slots to remove (partial Fisher-Yates)
        for a in range(kk):
            j = a + np.random.randint(na - a)
            tmp = rem[a]; rem[a] = rem[j]; rem[j] = tmp
        wi = wu[i]
        dsq = 0.0
        for a in range(kk):
            t = rem[a]
            x[i, t] = 0
            old = L[t]; L[t] = old - wi
            dsq += (float(L[t]) ** 2 - float(old) ** 2)
            seg_set(tree, size, t, L[t])
        # free slots in the window (after removal; may include removed ones)
        nf = 0
        for t in range(s[i], e[i] + 1):
            if x[i, t] == 0:
                add[nf] = t; nf += 1
        ka = kk if kk <= nf else nf
        for a in range(ka):
            j = a + np.random.randint(nf - a)
            tmp = add[a]; add[a] = add[j]; add[j] = tmp
        for a in range(ka):
            t = add[a]
            x[i, t] = 1
            old = L[t]; L[t] = old + wi
            dsq += (float(L[t]) ** 2 - float(old) ** 2)
            seg_set(tree, size, t, L[t])
        f_new = tree[1] / unit + eps * (sq + dsq / (unit * unit))
        delta = f_new - f_cur
        if delta < 0 or np.random.random() < np.exp(-delta / theta):
            f_cur = f_new; sq += dsq / (unit * unit)
        else:  # revert
            for a in range(ka):
                t = add[a]; x[i, t] = 0; L[t] -= wi; seg_set(tree, size, t, L[t])
            for a in range(kk):
                t = rem[a]; x[i, t] = 1; L[t] += wi; seg_set(tree, size, t, L[t])
        pk = tree[1]
        better = pk < best_pk or (eps > 0.0 and pk == best_pk and sq < best_sq - 1e-12)
        if better:
            if pk < best_pk and nrec < rec_it.shape[0]:
                rec_it[nrec] = k; rec_val[nrec] = pk; nrec += 1
            best_pk = pk; best_sq = sq
            best_x[:, :] = x
            best_L[:] = L
        k += 1; done += 1; ep += 1
        if ep >= ell:
            ep = 0; theta *= alpha_c
    state[0] = theta; state[1] = f_cur; state[2] = k; state[3] = ep; state[4] = best_pk; state[5] = best_sq
    return nrec


# ── systematic sampling and MT ─────────────────────────────────────────────────

@njit(cache=True)
def systematic_draw(c, r):
    """Indices selected by systematic sampling with cumulative cut points c (len m+1, c[-1] = r)."""
    out = np.empty(r, np.int64)
    u = np.random.random()
    m = c.shape[0] - 1
    j = 0
    for k in range(r):
        pt = u + k
        while j < m - 1 and c[j + 1] <= pt:
            j += 1
        out[k] = j
    return out


@njit(cache=True)
def mt_run(x0, L0, wu, fr_ptr, fr_slot, fr_cum, fr_r, veh_of, slot_ptr, slot_veh, u_units, max_steps):
    """Moser-Tardos resampling. x0/L0: schedule with the integral entries of alpha only.
    Random vehicles v = 0..V-1 (global id veh_of[v]) have fractional slots fr_slot[fr_ptr[v]:fr_ptr[v+1]]
    with cumulative cut points fr_cum[fr_ptr[v] + v : fr_ptr[v+1] + v + 1] and size fr_r[v].
    slot_veh[slot_ptr[t]:slot_ptr[t+1]] are the random vehicles (local ids) of slot t.
    Returns (x, L, number of resamplings, success)."""
    x = x0.copy(); L = L0.copy()
    V = fr_r.shape[0]; T = L.shape[0]
    for v in range(V):
        _mt_draw(v, x, L, wu, fr_ptr, fr_slot, fr_cum, fr_r, veh_of)
    steps = 0
    bad = np.empty(T, np.int64)
    while steps < max_steps:
        nb = 0
        for t in range(T):
            if L[t] > u_units:
                bad[nb] = t; nb += 1
        if nb == 0:
            return x, L, steps, True
        t = bad[np.random.randint(nb)]
        for q in range(slot_ptr[t], slot_ptr[t + 1]):
            v = slot_veh[q]; gi = veh_of[v]
            for z in range(fr_ptr[v], fr_ptr[v + 1]):
                tt = fr_slot[z]
                if x[gi, tt]:
                    x[gi, tt] = 0; L[tt] -= wu[gi]
            _mt_draw(v, x, L, wu, fr_ptr, fr_slot, fr_cum, fr_r, veh_of)
        steps += 1
    return x, L, steps, False


@njit(cache=True)
def _mt_draw(v, x, L, wu, fr_ptr, fr_slot, fr_cum, fr_r, veh_of):
    a = fr_ptr[v]; b = fr_ptr[v + 1]
    c = fr_cum[a + v: b + v + 1]
    sel = systematic_draw(c, fr_r[v])
    gi = veh_of[v]
    for q in range(sel.shape[0]):
        t = fr_slot[a + sel[q]]
        x[gi, t] = 1; L[t] += wu[gi]


# ── CE: derandomized rounding by conditional expectations ──────────────────────

@njit(cache=True)
def _conv_law(ws, as_, cnt, H):
    law = np.zeros(H + 1)
    law[0] = 1.0
    ln = 1
    for q in range(cnt):
        w = ws[q]; a = as_[q]
        for k in range(ln - 1, -1, -1):
            v = law[k]
            law[k] = v * (1.0 - a)
            law[k + w] += v * a
        ln += w
    return law, ln


@njit(cache=True)
def _remove_bern(law, ln, w, a):
    """Law of R' from the law of R' + w B(a) (stable forward/backward recursion)."""
    L2 = ln - w
    out = np.zeros(law.shape[0])
    if a <= 0.5:
        for k in range(L2):
            prev = out[k - w] if k >= w else 0.0
            out[k] = (law[k] - a * prev) / (1.0 - a)
    else:
        for k in range(ln - 1, w - 1, -1):
            hi = out[k] if k < L2 else 0.0
            out[k - w] = (law[k] - (1.0 - a) * hi) / a
    tot = 0.0
    for k in range(L2):
        if out[k] < 0.0:
            out[k] = 0.0
        tot += out[k]
    for k in range(L2):
        out[k] /= tot
    return out, L2


@njit(cache=True)
def _value(c, law, ln, u, excess):
    v = 0.0
    for k in range(ln):
        val = c + k
        if excess:
            if val > u:
                v += law[k] * (val - u)
        else:
            if val > u:
                v += law[k]
    return v


@njit(cache=True)
def ce_run(alpha, s, e, p, wu, u, excess, passes, compatible, tol):
    """Derandomized rounding. excess=True: estimator u + sum E[(S_t - u)^+]; False: sum P(S_t > u).
    compatible=True: vehicles keep the integral entries of alpha and choose only among their
    fractional slots (componentwise version). Returns (x, L)."""
    n, T = alpha.shape
    israndom = np.zeros((n, T), np.bool_)
    det = np.zeros(T, np.int64)
    cnt = np.zeros(T, np.int64)
    H = 0
    for t in range(T):
        h = 0
        for i in range(n):
            a = alpha[i, t]
            if a >= 1.0 - tol:
                det[t] += wu[i]
            elif a > tol:
                israndom[i, t] = True; cnt[t] += 1; h += wu[i]
        if h > H:
            H = h
    # slot laws (row t: law of the random part), lengths
    laws = np.zeros((T, H + 1))
    lens = np.ones(T, np.int64)
    ws = np.empty(n, np.int64); as_ = np.empty(n)
    for t in range(T):
        c = 0
        for i in range(n):
            if israndom[i, t]:
                ws[c] = wu[i]; as_[c] = alpha[i, t]; c += 1
        lw, ln = _conv_law(ws, as_, c, H)
        laws[t, :] = lw; lens[t] = ln
    x = np.zeros((n, T), np.int8)
    fixed = np.zeros(n, np.bool_)
    has_frac = np.zeros(n, np.bool_)
    for i in range(n):
        for t in range(s[i], e[i] + 1):
            if israndom[i, t]:
                has_frac[i] = True
            if alpha[i, t] >= 1.0 - tol:
                x[i, t] = 1
    if compatible:
        for i in range(n):
            if not has_frac[i]:
                fixed[i] = True
    order = np.argsort(-wu, kind="mergesort")
    nseq = n * (1 + passes)
    for q in range(nseq):
        i = order[q % n]
        if compatible and not has_frac[i]:
            continue
        wi = wu[i]
        m = e[i] - s[i] + 1
        dminus = np.empty(m, np.int64)
        delta = np.empty(m)
        row = np.full(m, -1, np.int64)
        nr = 0
        for k in range(m):
            if (not fixed[i]) and israndom[i, s[i] + k]:
                row[k] = nr; nr += 1
        newlaw = np.zeros((max(nr, 1), H + 1))
        newlen = np.zeros(max(nr, 1), np.int64)
        elig = np.ones(m, np.bool_)
        for k in range(m):
            t = s[i] + k
            if row[k] >= 0:
                if cnt[t] <= 12:
                    c = 0
                    for j in range(n):
                        if israndom[j, t] and j != i:
                            ws[c] = wu[j]; as_[c] = alpha[j, t]; c += 1
                    lw, ln = _conv_law(ws, as_, c, H)
                else:
                    lw, ln = _remove_bern(laws[t], lens[t], wi, alpha[i, t])
                newlaw[row[k], :] = lw; newlen[row[k]] = ln
                dminus[k] = det[t]
                delta[k] = _value(dminus[k] + wi, newlaw[row[k]], ln, u, excess) - _value(dminus[k], newlaw[row[k]], ln, u, excess)
            else:
                cur = x[i, t]
                dminus[k] = det[t] - wi * cur
                delta[k] = _value(dminus[k] + wi, laws[t], lens[t], u, excess) - _value(dminus[k], laws[t], lens[t], u, excess)
            if compatible:
                elig[k] = alpha[i, t] > tol and alpha[i, t] < 1.0 - tol
        # how many slots to choose among the eligible ones
        need = p[i]
        if compatible:
            for k in range(m):
                if not elig[k] and alpha[i, s[i] + k] >= 1.0 - tol:
                    need -= 1
        # lexicographic order: delta, then current load
        o1 = np.argsort(dminus, kind="mergesort")
        d1 = delta[o1]
        o2 = np.argsort(d1, kind="mergesort")
        on = np.zeros(m, np.bool_)
        got = 0
        for z in range(m):
            k = o1[o2[z]]
            if got >= need:
                break
            if elig[k]:
                on[k] = True; got += 1
        if compatible:
            for k in range(m):
                if not elig[k] and alpha[i, s[i] + k] >= 1.0 - tol:
                    on[k] = True
        for k in range(m):
            t = s[i] + k
            det[t] = dminus[k] + (wi if on[k] else 0)
            if row[k] >= 0:
                laws[t, :] = newlaw[row[k]]; lens[t] = newlen[row[k]]
                israndom[i, t] = False; cnt[t] -= 1
            x[i, t] = 1 if on[k] else 0
        fixed[i] = True
    return x, loads_of(x, wu)


# ── LLS: load-levelling iterated local search ──────────────────────────────────

@njit(cache=True)
def _descent(x, L, s, e, wu, order):
    """Unit moves t -> t' with L[t'] + w_i < L[t] until none applies. Returns the number of moves."""
    n = x.shape[0]
    moves = 0
    changed = True
    while changed:
        changed = False
        for oi in range(n):
            i = order[oi]
            wi = wu[i]
            while True:
                t_on = -1; t_off = -1; von = -BIG; voff = BIG
                for t in range(s[i], e[i] + 1):
                    if x[i, t]:
                        if L[t] > von:
                            von = L[t]; t_on = t
                    else:
                        if L[t] < voff:
                            voff = L[t]; t_off = t
                if t_on < 0 or t_off < 0 or voff + wi >= von:
                    break
                x[i, t_on] = 0; x[i, t_off] = 1
                L[t_on] -= wi; L[t_off] += wi
                moves += 1; changed = True
    return moves


@njit(cache=True)
def _ejection(x, L, s, e, wu, max_checks):
    """One two-step chain i: t* -> t1, j: t1 -> t2 relieving a peak slot without creating a new peak."""
    n, T = x.shape
    peak = L.max()
    checks = 0
    for ts in range(T):
        if L[ts] != peak:
            continue
        for i in range(n):
            if x[i, ts] == 0:
                continue
            wi = wu[i]
            for t1 in range(s[i], e[i] + 1):
                if x[i, t1] or L[t1] + wi < peak:
                    continue
                need = L[t1] + wi - peak
                for j in range(n):
                    checks += 1
                    if checks > max_checks:
                        return False
                    if j == i or x[j, t1] == 0 or wu[j] <= need:
                        continue
                    wj = wu[j]
                    best = -1; bv = BIG
                    for t2 in range(s[j], e[j] + 1):
                        if x[j, t2] == 0 and t2 != t1 and t2 != ts and L[t2] + wj < peak and L[t2] < bv:
                            bv = L[t2]; best = t2
                    if best >= 0:
                        x[i, ts] = 0; x[i, t1] = 1; x[j, t1] = 0; x[j, best] = 1
                        L[ts] -= wi; L[t1] += wi - wj; L[best] += wj
                        return True
    return False


@njit(cache=True)
def _better_or_equal(La, Lb):
    pa, ca = count_peak(La); pb, cb = count_peak(Lb)
    if pa != pb:
        return pa < pb
    if ca != cb:
        return ca < cb
    return sum_sq(La) <= sum_sq(Lb)


@njit(cache=True)
def lls_local(x, L, s, e, wu, max_checks):
    n = x.shape[0]
    order = np.random.permutation(n)
    _descent(x, L, s, e, wu, order)
    while _ejection(x, L, s, e, wu, max_checks):
        _descent(x, L, s, e, wu, order)


@njit(cache=True)
def lls_iterate(best_x, best_L, s, e, wu, n_kicks, kick_size, max_checks, lb_units, rec_it, rec_val, it0):
    """n_kicks iterations of: kick the best schedule, local search, accept if not worse (lexicographic)."""
    n, T = best_x.shape
    nrec = 0
    cand = np.empty(n, np.int64)
    for it in range(n_kicks):
        if best_L.max() <= lb_units:
            break
        y = best_x.copy(); Ly = best_L.copy()
        pk = Ly.max()
        nc = 0
        for i in range(n):
            for t in range(s[i], e[i] + 1):
                if y[i, t] and Ly[t] == pk:
                    cand[nc] = i; nc += 1
                    break
        for q in range(kick_size):
            if nc > 0 and q % 2 == 0:
                i = cand[np.random.randint(nc)]
            else:
                i = np.random.randint(n)
            non = 0; noff = 0
            for t in range(s[i], e[i] + 1):
                if y[i, t]:
                    non += 1
                else:
                    noff += 1
            if non == 0 or noff == 0:
                continue
            a = np.random.randint(non); b = np.random.randint(noff)
            ta = -1; tb = -1
            for t in range(s[i], e[i] + 1):
                if y[i, t]:
                    if a == 0 and ta < 0:
                        ta = t
                    a -= 1
                else:
                    if b == 0 and tb < 0:
                        tb = t
                    b -= 1
            y[i, ta] = 0; y[i, tb] = 1; Ly[ta] -= wu[i]; Ly[tb] += wu[i]
        lls_local(y, Ly, s, e, wu, max_checks)
        if _better_or_equal(Ly, best_L):
            if Ly.max() < best_L.max() and nrec < rec_it.shape[0]:
                rec_it[nrec] = it0 + it; rec_val[nrec] = Ly.max(); nrec += 1
            best_x[:, :] = y; best_L[:] = Ly
    return nrec
