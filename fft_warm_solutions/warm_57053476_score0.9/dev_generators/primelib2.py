# component-phased prime DFT with k-splits; in-place-safe via SC stash.
from netlib import E, KC, fmt, trigc, trigs

KSPLITS = {13: [[1,2,3,4,5,6]], 17: [[1,2,3,4],[5,6,7,8]], 23: [[1,2,3,4,5,6],[7,8,9,10,11]]}

def emit_prime_reim(e, p, loader, storer, sc, ksplits=None):
    """loader(e, j) -> (re_expr, im_expr) must RE-EMIT loads on each call (memory source).
       storer(e, k, re, im). sc: scratch symbol with >= (p+3)*16 doubles.
       SC layout: per k (1..h): Cr at (k-1)*16, Sr at +8 ; y0r at h*16; stash blocks after (p)*16."""
    h = (p-1)//2
    if ksplits is None: ksplits = KSPLITS[p]
    cosv = {r: e.v(KC(trigc(r, p))) for r in range(1, h+1)}
    sinv = {r: e.v(KC(trigs(r, p))) for r in range(1, h+1)}
    nblk = len(ksplits)
    stash_base = (h+1)*16
    for comp in (0, 1):
        first_blk = True
        for bi, ks in enumerate(ksplits):
            x0 = loader(e, 0)[comp]
            C = {k: x0 for k in ks}
            S = {k: None for k in ks}
            y0 = x0
            for j in range(1, h+1):
                pair = loader(e, j), loader(e, p-j)
                a = pair[0][comp]; b = pair[1][comp]
                er = e.v(f"{a} + {b}")
                orr = e.v(f"{a} - {b}")
                if first_blk:
                    y0 = e.v(f"{y0} + {er}")
                for k in ks:
                    rr0 = (k*j) % p
                    rr = rr0 if rr0 <= h else p - rr0
                    ssign = 1 if rr0 <= h else -1
                    C[k] = e.v(f"FMA({cosv[rr]}, {er}, {C[k]})")
                    F = "FMA" if ssign > 0 else "FNMA"
                    if S[k] is None:
                        S[k] = e.v(f"{sinv[rr]} * {orr}") if ssign > 0 else e.v(f"-({sinv[rr]} * {orr})")
                    else:
                        S[k] = e.v(f"{F}({sinv[rr]}, {orr}, {S[k]})")
            if comp == 0:
                for k in ks:
                    e.raw(f"ST({sc} + {(k-1)*16}, {C[k]}); ST({sc} + {(k-1)*16+8}, {S[k]});")
                if first_blk:
                    e.raw(f"ST({sc} + {h*16}, {y0});")
            else:
                is_last = (bi == nblk-1)
                if first_blk:
                    y0r = e.v(f"LD({sc} + {h*16})")
                    if is_last:
                        storer(e, 0, y0r, y0)
                    else:
                        e.raw(f"ST({sc} + {h*16}, {y0r}); ST({sc} + {h*16+8}, {y0});")
                for k in ks:
                    cr = e.v(f"LD({sc} + {(k-1)*16})"); sr = e.v(f"LD({sc} + {(k-1)*16+8})")
                    ci, si = C[k], S[k]
                    xr1 = e.v(f"{cr} - {si}"); xi1 = e.v(f"{ci} + {sr}")
                    xr2 = e.v(f"{cr} + {si}"); xi2 = e.v(f"{ci} - {sr}")
                    if is_last:
                        storer(e, k, xr1, xi1)
                        storer(e, p-k, xr2, xi2)
                    else:
                        o = stash_base + (k-1)*32
                        e.raw(f"ST({sc} + {o}, {xr1}); ST({sc} + {o+8}, {xi1});")
                        e.raw(f"ST({sc} + {o+16}, {xr2}); ST({sc} + {o+24}, {xi2});")
            first_blk = False
    # flush stashed blocks (all but last) + y0 if stashed
    if nblk > 1:
        e.raw("BAR();")
        y0r = e.v(f"LD({sc} + {h*16})"); y0i = e.v(f"LD({sc} + {h*16+8})")
        storer(e, 0, y0r, y0i)
        for ks in ksplits[:-1]:
            for k in ks:
                o = stash_base + (k-1)*32
                r1 = e.v(f"LD({sc} + {o})"); i1 = e.v(f"LD({sc} + {o+8})")
                r2 = e.v(f"LD({sc} + {o+16})"); i2 = e.v(f"LD({sc} + {o+24})")
                storer(e, k, r1, i1)
                storer(e, p-k, r2, i2)
