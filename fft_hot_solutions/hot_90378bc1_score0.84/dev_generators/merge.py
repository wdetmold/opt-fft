import re, sys

prior = open('/tmp/g/vendored_prior.c').read()
f3f30 = open('/tmp/g/vendored_3f30.c').read()
mine  = open('/tmp/g/impl_fg.c').read()

# prefix my generated engine's global identifiers
mine_ids = ['huge_alloc','init_36','init_45','run_36','run_45','bench_36','bench_45',
            'convin_36','convin_45','convc_36','convc_45','convout_36','convout_45',
            'phaseA_36','phaseA_45','phaseB_36','phaseB_45',
            'S36_re','S36_im','S45_re','S45_im','CT36_re','CT36_im','CT45_re','CT45_im',
            'SPOS36','SPOS45','TPOS36','TPOS45','FS36','FS45','FT36','FT45',
            'SCR36_re','SCR36_im','SCR45_re','SCR45_im','bcastv','N36','N45']
for ident in mine_ids:
    mine = re.sub(r'\b' + ident + r'\b', 'MJ_' + ident, mine)

out = []
out.append('// ============================================================\n'
           '// implementation.c -- merged single-file build\n'
           '// Section 1: vendored engine A (prior-work 00291a90 final: all sizes;\n'
           '//            used for L=6,8,13,17,23,45)\n'
           '// Section 2: vendored engine B (prior-work v6_3f30d81f: used for L=64)\n'
           '// Section 3: this round\'s 6D-PFA digit engine (L=36; also provides 45)\n'
           '// ============================================================\n')
out.append(prior)
out.append('\n// ================= SECTION 2: engine B (3f30) =================\n')
out.append(f3f30)
out.append('\n// ================= SECTION 3: engine C (PFA digit) =================\n')
out.append(mine)
open('/tmp/g/implementation.c','w').write('\n'.join(out))
print('merged', sum(len(x) for x in out))
