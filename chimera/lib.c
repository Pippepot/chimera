#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef void (*fmt_func_t)(char* buf, size_t buf_size, const char* format, const void* elem);

void i_fmt(char* buf, size_t buf_size, const char* format, const void* elem) {
    sprintf(buf, format, *((const int*)elem));
}
void f_fmt(char* buf, size_t buf_size, const char* format, const void* elem) {
    sprintf(buf, format, *((const float*)elem));
}

static void build_string(char* buf, size_t* pos, char* base, int dim, int dims,
                         int* shape, int* strides, size_t item_size,
                         const char* format, fmt_func_t formatter) {
    buf[(*pos)++] = '[';
    for (int i = 0; i < shape[dim]; i++) {
        if (i > 0) {
            buf[(*pos)++] = ',';
            buf[(*pos)++] = ' ';
        }
        char* cur = base + i * strides[dim] * item_size;
        if (dim == dims - 1) {
            char temp[16];
            formatter(temp, sizeof(temp), format, cur);
            size_t len = strlen(temp);
            memcpy(buf + *pos, temp, len);
            *pos += len;
        } else {
            build_string(buf, pos, cur, dim + 1, dims, shape, strides, item_size, format, formatter);
        }
    }
    buf[(*pos)++] = ']';
}

char* array_to_string(void* p, size_t item_size, size_t total_size,
                      int* shape, int dims, int* strides,
                      const char* format, fmt_func_t formatter) {
    // rough estimate
    size_t buf_size = total_size * 16 + dims * 2 + total_size * 2 + 1;
    char* result = malloc(buf_size);
    if (!result)
        return NULL;
    
    size_t pos = 0;
    build_string(result, &pos, (char*)p, 0, dims, shape, strides, item_size, format, formatter);
    result[pos] = '\0';
    return result;
}