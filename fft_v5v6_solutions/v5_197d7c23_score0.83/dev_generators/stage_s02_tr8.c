
// ---------------------------------------------------------------- 8x8 transpose of v8 rows (in-place)
AI void tr8(v8 *a){
    v8 u0=__builtin_shufflevector(a[0],a[1],0,8,2,10,4,12,6,14);
    v8 u1=__builtin_shufflevector(a[0],a[1],1,9,3,11,5,13,7,15);
    v8 u2=__builtin_shufflevector(a[2],a[3],0,8,2,10,4,12,6,14);
    v8 u3=__builtin_shufflevector(a[2],a[3],1,9,3,11,5,13,7,15);
    v8 u4=__builtin_shufflevector(a[4],a[5],0,8,2,10,4,12,6,14);
    v8 u5=__builtin_shufflevector(a[4],a[5],1,9,3,11,5,13,7,15);
    v8 u6=__builtin_shufflevector(a[6],a[7],0,8,2,10,4,12,6,14);
    v8 u7=__builtin_shufflevector(a[6],a[7],1,9,3,11,5,13,7,15);
    v8 w0=__builtin_shufflevector(u0,u2,0,1,8,9,4,5,12,13);
    v8 w1=__builtin_shufflevector(u1,u3,0,1,8,9,4,5,12,13);
    v8 w2=__builtin_shufflevector(u0,u2,2,3,10,11,6,7,14,15);
    v8 w3=__builtin_shufflevector(u1,u3,2,3,10,11,6,7,14,15);
    v8 w4=__builtin_shufflevector(u4,u6,0,1,8,9,4,5,12,13);
    v8 w5=__builtin_shufflevector(u5,u7,0,1,8,9,4,5,12,13);
    v8 w6=__builtin_shufflevector(u4,u6,2,3,10,11,6,7,14,15);
    v8 w7=__builtin_shufflevector(u5,u7,2,3,10,11,6,7,14,15);
    a[0]=__builtin_shufflevector(w0,w4,0,1,2,3,8,9,10,11);
    a[4]=__builtin_shufflevector(w0,w4,4,5,6,7,12,13,14,15);
    a[1]=__builtin_shufflevector(w1,w5,0,1,2,3,8,9,10,11);
    a[5]=__builtin_shufflevector(w1,w5,4,5,6,7,12,13,14,15);
    a[2]=__builtin_shufflevector(w2,w6,0,1,2,3,8,9,10,11);
    a[6]=__builtin_shufflevector(w2,w6,4,5,6,7,12,13,14,15);
    a[3]=__builtin_shufflevector(w3,w7,0,1,2,3,8,9,10,11);
    a[7]=__builtin_shufflevector(w3,w7,4,5,6,7,12,13,14,15);
}

// ---------------------------------------------------------------- buffers
