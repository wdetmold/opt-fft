src = open('solution.py').read()
src = src.replace("""    subprocess.run(['gcc', '-O3', '-march=native', '-ffp-contract=fast',
                    '-fno-schedule-insns', '-fno-math-errno', '-fno-trapping-math',
                    '-shared', '-fPIC', _c, '-o', _so, '-lm'], check=True)   # add flags if needed""",
"""    subprocess.run(['gcc', '-O3', '-march=native', '-ffp-contract=fast',
                    '-fno-schedule-insns', '-fno-math-errno', '-fno-trapping-math',
                    '-falign-loops=32', '-falign-functions=64',
                    '-shared', '-fPIC', _c, '-o', _so, '-lm'], check=True)   # add flags if needed""")
open('solution.py','w').write(src)
