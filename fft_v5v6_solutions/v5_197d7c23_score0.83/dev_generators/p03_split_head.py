# S03 (log 01:17:13): split implementation.c -> impl_head.c at the buffers marker
src = open('implementation.c').read()
cut = src.index('// ---------------------------------------------------------------- buffers')
open('impl_head.c','w').write(src[:cut])
print("head lines:", src[:cut].count('\n'))
