cd /workdir/dev && python3 - <<'EOF'
src = open('gen.py').read()
# sq_yz: double-buffer: use buf index cc&1
old = """static void sq{L}_yz(V* sr, V* si) {{
  for (long cc = 0; cc < {NB}; cc++) {{
    f{L}_cb(sr + cc, si + cc, sq{L}_buf_r, sq{L}_buf_i);
    for (long rb = 0; rb < {P1V}; rb++) {{
      V tb[8];
      tr8x8(sq{L}_buf_r + rb*8, tb);
      for (int q = 0; q < 8; q++) slabts{L}_r[(cc*8+q)*{P1V} + rb] = tb[q];
      tr8x8(sq{L}_buf_i + rb*8, tb);
      for (int q = 0; q < 8; q++) slabts{L}_i[(cc*8+q)*{P1V} + rb] = tb[q];
    }}
  }}
  for (long cc = 0; cc < {NB}; cc++) {{
    f{L}_cb(slabts{L}_r + cc, slabts{L}_i + cc, sq{L}_buf_r, sq{L}_buf_i);
    int lim = (int)({L} - cc*8); if (lim > 8) lim = 8;
    for (long rb = 0; rb < {P1V}; rb++) {{
      V tb[8];
      tr8x8(sq{L}_buf_r + rb*8, tb);
      for (int q = 0; q < lim; q++) sr[(cc*8+q)*{P1V} + rb] = tb[q];
      tr8x8(sq{L}_buf_i + rb*8, tb);
      for (int q = 0; q < lim; q++) si[(cc*8+q)*{P1V} + rb] = tb[q];
    }}
  }}
}}"""
new = """static void sq{L}_yz(V* sr, V* si) {{
  for (long cc = 0; cc < {NB}; cc++) {{
    V *br_ = sq{L}_buf_r + (cc&1)*{P1}, *bi_ = sq{L}_buf_i + (cc&1)*{P1};
    f{L}_cb(sr + cc, si + cc, br_, bi_);
    for (long rb = 0; rb < {P1V}; rb++) {{
      V tb[8];
      tr8x8(br_ + rb*8, tb);
      for (int q = 0; q < 8; q++) slabts{L}_r[(cc*8+q)*{P1V} + rb] = tb[q];
      tr8x8(bi_ + rb*8, tb);
      for (int q = 0; q < 8; q++) slabts{L}_i[(cc*8+q)*{P1V} + rb] = tb[q];
    }}
  }}
  for (long cc = 0; cc < {NB}; cc++) {{
    V *br_ = sq{L}_buf_r + (cc&1)*{P1}, *bi_ = sq{L}_buf_i + (cc&1)*{P1};
    f{L}_cb(slabts{L}_r + cc, slabts{L}_i + cc, br_, bi_);
    int lim = (int)({L} - cc*8); if (lim > 8) lim = 8;
    for (long rb = 0; rb < {P1V}; rb++) {{
      V tb[8];
      tr8x8(br_ + rb*8, tb);
      for (int q = 0; q < lim; q++) sr[(cc*8+q)*{P1V} + rb] = tb[q];
      tr8x8(bi_ + rb*8, tb);
      for (int q = 0; q < lim; q++) si[(cc*8+q)*{P1V} + rb] = tb[q];
    }}
  }}
}}"""
assert old in src
src = src.replace(old, new)
src = src.replace("static V sq{L}_buf_r[{P1}], sq{L}_buf_i[{P1}];", "static V sq{L}_buf_r[2*{P1}], sq{L}_buf_i[2*{P1}];")
# zero both buffers' tails
src = src.replace("""  for (long j = {L}; j < {P1}; j++) {{ sq{L}_buf_r[j] = VC(0.0); sq{L}_buf_i[j] = VC(0.0); }}""",
"""  for (long j = {L}; j < {P1}; j++) {{ sq{L}_buf_r[j] = VC(0.0); sq{L}_buf_i[j] = VC(0.0);
    sq{L}_buf_r[{P1}+j] = VC(0.0); sq{L}_buf_i[{P1}+j] = VC(0.0); }}""")
# same for slab scheme
old2 = """      for (long cc = 0; cc < {NB}; cc++) {{
        f{L}_cb(sr + cc, si + cc, slab{L}_br, slab{L}_bi);
        for (long rb = 0; rb < {P1V}; rb++) {{
          V tb[8];
          tr8x8(slab{L}_br + rb*8, tb);
          for (int q = 0; q < 8; q++) slab{L}_tsr[(cc*8+q)*{P1V} + rb] = tb[q];
          tr8x8(slab{L}_bi + rb*8, tb);
          for (int q = 0; q < 8; q++) slab{L}_tsi[(cc*8+q)*{P1V} + rb] = tb[q];
        }}
      }}
      for (long cc = 0; cc < {NB}; cc++) {{
        f{L}_cb(slab{L}_tsr + cc, slab{L}_tsi + cc, slab{L}_br, slab{L}_bi);
        int lim = (int)({L} - cc*8); if (lim > 8) lim = 8;
        for (long rb = 0; rb < {P1V}; rb++) {{
          V tb[8];
          tr8x8(slab{L}_br + rb*8, tb);
          for (int q = 0; q < lim; q++) sr[(cc*8+q)*{P1V} + rb] = tb[q];
          tr8x8(slab{L}_bi + rb*8, tb);
          for (int q = 0; q < lim; q++) si[(cc*8+q)*{P1V} + rb] = tb[q];
        }}
      }}"""
new2 = """      for (long cc = 0; cc < {NB}; cc++) {{
        V *br_ = slab{L}_br + (cc&1)*{P1}, *bi_ = slab{L}_bi + (cc&1)*{P1};
        f{L}_cb(sr + cc, si + cc, br_, bi_);
        for (long rb = 0; rb < {P1V}; rb++) {{
          V tb[8];
          tr8x8(br_ + rb*8, tb);
          for (int q = 0; q < 8; q++) slab{L}_tsr[(cc*8+q)*{P1V} + rb] = tb[q];
          tr8x8(bi_ + rb*8, tb);
          for (int q = 0; q < 8; q++) slab{L}_tsi[(cc*8+q)*{P1V} + rb] = tb[q];
        }}
      }}
      for (long cc = 0; cc < {NB}; cc++) {{
        V *br_ = slab{L}_br + (cc&1)*{P1}, *bi_ = slab{L}_bi + (cc&1)*{P1};
        f{L}_cb(slab{L}_tsr + cc, slab{L}_tsi + cc, br_, bi_);
        int lim = (int)({L} - cc*8); if (lim > 8) lim = 8;
        for (long rb = 0; rb < {P1V}; rb++) {{
          V tb[8];
          tr8x8(br_ + rb*8, tb);
          for (int q = 0; q < lim; q++) sr[(cc*8+q)*{P1V} + rb] = tb[q];
          tr8x8(bi_ + rb*8, tb);
          for (int q = 0; q < lim; q++) si[(cc*8+q)*{P1V} + rb] = tb[q];
        }}
      }}"""
assert old2 in src
src = src.replace(old2, new2)
src = src.replace("static V slab{L}_br[{P1}], slab{L}_bi[{P1}];", "static V slab{L}_br[2*{P1}], slab{L}_bi[2*{P1}];")
src = src.replace("""  for (long j = {L}; j < {P1}; j++) {{ slab{L}_br[j] = VC(0.0); slab{L}_bi[j] = VC(0.0); }}""",
"""  for (long j = {L}; j < {P1}; j++) {{ slab{L}_br[j] = VC(0.0); slab{L}_bi[j] = VC(0.0);
    slab{L}_br[{P1}+j] = VC(0.0); slab{L}_bi[{P1}+j] = VC(0.0); }}""")
open('gen.py','w').write(src)
print("double-buffered")
EOF
python3 -c "
import tune
cfg={'pw':{'13':'alt','17':'alt','23':'alt','45':'alt'}}
so=tune.build(cfg,'db')
r=tune.bench(so)
print({L: round(v,2) for L,v in r.items()})
"