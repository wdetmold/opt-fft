cd /workdir && gcc -O1 -g -march=native -fsanitize=address -shared -fPIC implementation.c -o /tmp/impl_asan.so -lm 2>&1 | head -3 && python3 - <<'EOF'
import os
os.environ['LD_PRELOAD'] = '/usr/lib/gcc/x86_64-linux-gnu/13/libasan.so' if os.path.exists('/usr/lib/gcc/x86_64-linux-gnu/13/libasan.so') else ''
print("preload:", os.environ['LD_PRELOAD'])
EOF
ls /usr/lib/x86_64-linux-gnu/libasan* /usr/lib/gcc/x86_64-linux-gnu/13/libasan* 2>/dev/null