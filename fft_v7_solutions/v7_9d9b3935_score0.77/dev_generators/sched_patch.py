import re
src = open('gen.py').read()

# 1) Add scheduling option to emit_codelet: level-order interleave
old_visit = '''    # topo order over reachable nodes
    roots = [i for xy in X for i in xy]
    order = []
    state = {}
    def visit(i):
        st = state.get(i, 0)
        if st == 2: return
        if st == 1: raise RuntimeError("cycle")
        state[i] = 1
        t = d.nodes[i]
        for ch in t[1:]:
            if isinstance(ch, int): visit(ch)
        state[i] = 2
        order.append(i)
    for r in roots: visit(r)'''
new_visit = '''    # topo order over reachable nodes
    roots = [i for xy in X for i in xy]
    order = topo_order(d, roots)'''
assert old_visit in src
src = src.replace(old_visit, new_visit)

# add topo_order function with SCHED mode
helper = '''
SCHED = 'level'
def topo_order(d, roots):
    order = []
    state = {}
    def visit(i):
        st = state.get(i, 0)
        if st == 2: return
        if st == 1: raise RuntimeError("cycle")
        state[i] = 1
        t = d.nodes[i]
        for ch in t[1:]:
            if isinstance(ch, int): visit(ch)
        state[i] = 2
        order.append(i)
    for r in roots: visit(r)
    if SCHED == 'dfs':
        return order
    # level order: depth = longest path from input
    depth = {}
    for i in order:
        t = d.nodes[i]
        ch = [c for c in t[1:] if isinstance(c, int)]
        depth[i] = 0 if not ch else 1 + max(depth[c] for c in ch)
    order2 = sorted(order, key=lambda i: (depth[i], order.index(i)))
    return order2

'''
src = src.replace('WIDTHS = {', helper + 'WIDTHS = {')
open('gen.py','w').write(src)
print("patched")
