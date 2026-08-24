cd /tmp/bench && python3 - <<'EOF'
# extract tables from the object file .rodata
import struct
fn='src_distributions_distributions.c.o'
data=open(fn,'rb').read()
ro_off=0x4480
def get(off, n=0x800):
    s=data[ro_off+off: ro_off+off+n]
    return s
wi = struct.unpack('<256d', get(0x3800))
ki = struct.unpack('<256Q', get(0x4000))
fi = struct.unpack('<256d', get(0x3000))
print("wi[0:3]:", wi[0:3]); print("ki[0:3]:", ki[0:3]); print("fi[0:3]:", fi[0:3]); print("fi[255]:", fi[255], "wi[255]:", wi[255], "ki[255]:", ki[255])
import pickle
pickle.dump({'wi':wi,'ki':ki,'fi':fi}, open('/tmp/bench/zig_tables.pkl','wb'))
# also hex forms
print(float.hex(wi[0]), float.hex(fi[255]))
EOF