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

char* array_to_string(void* p, size_t item_size, int* shape, int dims,
                      const char* format, fmt_func_t formatter) {
    int total = 1;
    for (int i = 0; i < dims; i++) {
        total *= shape[i];
    }
    
    int* strides = malloc(sizeof(int) * dims);
    if (!strides) return NULL;

    strides[dims - 1] = 1;
    for (int i = dims - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    // Rough buffer size: 
    // Assume <=16 chars per element + 2 chars per bracket comma + 1 for '\0'
    size_t buf_size = (size_t)total * 16 + dims * 2 + 1;
    char* buf = malloc(buf_size);
    if (!buf) {
        free(strides);
        return NULL;
    }

    size_t pos = 0;
    build_string(buf, &pos, (char*)p, 0, dims, shape, strides, item_size, format, formatter);
    buf[pos] = '\0';
    return buf;
}