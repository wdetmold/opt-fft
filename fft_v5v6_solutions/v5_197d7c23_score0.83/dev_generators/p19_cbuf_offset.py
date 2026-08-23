# B25 (log 02:35:23): offset CBUF to decorrelate cache sets vs SBUF
src=open('impl_tail.c').read()
old='''void init_buffers(void){
    SBUF=(v8*)xalloc((size_t)MAXBLK*128);
    CBUF=(v8*)xalloc((size_t)MAXBLK*128);
}'''
new='''void init_buffers(void){
    SBUF=(v8*)xalloc((size_t)MAXBLK*128);
    CBUF=(v8*)((char*)xalloc((size_t)MAXBLK*128 + 65536) + 2176+4096);  /* decorrelate sets vs SBUF */
}'''
assert old in src
src=src.replace(old,new)
open('impl_tail.c','w').write(src)
