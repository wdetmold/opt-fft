cd /workdir && python3 -c "
import sys; sys.path.insert(0,'/workdir/dev'); sys.path.insert(0,'/workdir')
from check import check
print('m=1:'); check(3, (3,3,3,3,3,2,2,2), (1,1,1,1,1,1,1,1), verbose=False)
print('m=2:'); check(4, (9,2,3,3,2,2,1,2), (2,2,2,2,2,2,2,2), verbose=False)
print('m=3:'); check(5, (1,1,1,1,1,1,1,3), (3,3,3,3,3,3,3,3), verbose=False)
" 2>/dev/null || (cd /workdir/dev && python3 -c "
import sys; sys.path.insert(0,'/workdir/dev'); sys.path.insert(0,'/workdir')
import check as c
print('m=1:'); c.check(3, (3,3,3,3,3,2,2,2), (1,1,1,1,1,1,1,1), verbose=False)
print('m=2:'); c.check(4, (9,2,3,3,2,2,1,2), (2,2,2,2,2,2,2,2), verbose=False)
print('m=3:'); c.check(5, (1,1,1,1,1,1,1,3), (3,3,3,3,3,3,3,3), verbose=False)
")