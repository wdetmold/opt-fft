#!/bin/bash
# Rebuild implementation.c exactly as the graded attempt did (final concatenation order).
cd "$(dirname "$0")"
cat impl_head.c impl_kern.c impl_body.c impl_fns.c impl_fns2.c impl_fns3.c impl_fns4.c impl_io.c impl_vl.c impl_p64.c impl_p4536.c impl_drv.c > ../implementation.c.rebuilt
echo "wrote ../implementation.c.rebuilt"
