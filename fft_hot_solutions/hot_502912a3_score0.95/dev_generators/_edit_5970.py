src = open('assemble.py').read()
src = src.replace("""    for nm in ('ALIGN64', 'TR8', 'VHALF', 'VONE', 'alloc_huge', 'alloc_huge_st', 'map2'):""",
                  """    for nm in ('ALIGN64', 'TR8', 'VHALF', 'VONE', 'alloc_huge', 'alloc_huge_st', 'map2', 'stagger_ctr', 'VONE_', 'VHALF_'):""")
open('assemble.py','w').write(src)
