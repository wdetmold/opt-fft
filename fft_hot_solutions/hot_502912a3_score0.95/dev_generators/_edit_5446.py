src = open('assemble.py').read()
src = src.replace("""    s39 = re.sub(r'^void run_(\\d+)\\(', r'static void s39_run_\\1(', s39, flags=re.M)""",
                  """    s39 = re.sub(r'^void run_(\\d+)\\(', r'static void s39_run_\\1(', s39, flags=re.M)
    s39 = re.sub(r'^static void run_(\\d+)\\(', r'static void s39_run_\\1(', s39, flags=re.M)""")
open('assemble.py','w').write(src)
