cd /workdir/dev && python3 - <<'EOF'
src = open('gen.py').read()
src = src.replace("_SCHEME_DEFAULT = {6:'soa', 8:'soa', 13:'soa', 17:'soa', 23:'soa', 36:'sq', 45:'apfa', 64:'sq'}",
                  "_SCHEME_DEFAULT = {6:'soa', 8:'soa', 13:'soa', 17:'soa', 23:'soa', 36:'sq', 45:'slab', 64:'sq'}")
old = """    const V *pcr = (it & 1) ? slab{L}_cer : slab{L}_cor;
    const V *pci = (it & 1) ? slab{L}_cei : slab{L}_coi;
    for (long o = 0; o < {L*P1V}; o++)
      f{L}_k3(slab{L}_xr + o, slab{L}_xi + o, slab{L}_xr + o, slab{L}_xi + o,
              pcr + o, pci + o);"""
# (note: slab driver was changed to always-natural single c earlier; find actual text)
print(old in src)
EOF
grep -n "f{L}_k3(slab" gen.py && sed -n '/for (long o = 0; o < {L\*P1V}; o++)/,+3p' gen.py