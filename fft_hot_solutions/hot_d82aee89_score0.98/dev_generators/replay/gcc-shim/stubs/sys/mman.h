#ifndef _STUB_SYS_MMAN_H
#define _STUB_SYS_MMAN_H
typedef __SIZE_TYPE__ size_t;
#define PROT_READ 0x1
#define PROT_WRITE 0x2
#define MAP_PRIVATE 0x02
#define MAP_ANONYMOUS 0x20
#define MAP_HUGETLB 0x40000
#define MAP_FAILED ((void*)-1)
#define MADV_HUGEPAGE 14
#define MADV_NOHUGEPAGE 15
void* mmap(void*, size_t, int, int, int, long);
int munmap(void*, size_t);
int madvise(void*, size_t, int);
#endif
