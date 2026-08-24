cd /workdir && cat > /tmp/asan_min.py <<'EOF'
import ctypes, numpy as np
lib = ctypes.CDLL('/tmp/impl_asan.so')
f=getattr(lib,'run6')
f.restype=None
f.argtypes=[ctypes.c_void_p]*2+[ctypes.c_long]*2+[ctypes.c_void_p]*2
B,m=1,1
n=B*216
x0=(np.arange(2*n)*0.001).astype(np.float64)
c=(np.arange(2*n)*0.0001).astype(np.float64)
o1=np.empty(2*n); om=np.empty(2*n)
print("calling run6 B=1 m=1", flush=True)
f(x0.ctypes.data,c.ctypes.data,B,m,o1.ctypes.data,om.ctypes.data)
print("OK", flush=True)
EOF
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libasan.so.8 ASAN_OPTIONS=detect_leaks=0 python3 /tmp/asan_min.py 2>&1 | grep -E "calling|OK|ERROR|#0|#1|#2|SUMMARY" | head -8