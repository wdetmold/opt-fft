#ifndef _STUB_STRING_H
#define _STUB_STRING_H
typedef __SIZE_TYPE__ size_t;
void* memcpy(void*, const void*, size_t);
void* memmove(void*, const void*, size_t);
void* memset(void*, int, size_t);
int memcmp(const void*, const void*, size_t);
size_t strlen(const char*);
int strcmp(const char*, const char*);
char* strcpy(char*, const char*);
#endif
