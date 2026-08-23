#ifndef _STUB_STDLIB_H
#define _STUB_STDLIB_H
typedef __SIZE_TYPE__ size_t;
void* malloc(size_t);
void* calloc(size_t, size_t);
void* realloc(void*, size_t);
void free(void*);
void* aligned_alloc(size_t, size_t);
int posix_memalign(void**, size_t, size_t);
void exit(int);
void abort(void);
int atoi(const char*);
long atol(const char*);
double atof(const char*);
#endif
