#ifndef _STUB_STDIO_H
#define _STUB_STDIO_H
typedef __SIZE_TYPE__ size_t;
typedef struct _FILE FILE;
extern FILE* stdout; extern FILE* stderr;
int printf(const char*, ...);
int fprintf(FILE*, const char*, ...);
int snprintf(char*, size_t, const char*, ...);
int puts(const char*);
int fflush(FILE*);
#endif
