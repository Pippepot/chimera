#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef void (*fmt_func_t)(char* buf, size_t buf_size, const void* elem);

void i_fmt(char* buf, size_t buf_size, const void* elem) {
    sprintf(buf, "%i", *((const int*)elem));
}
void f_fmt(char* buf, size_t buf_size, const void* elem) {
    sprintf(buf, "%f", *((const float*)elem));
}
void b_fmt(char* buf, size_t buf_size, const void* elem) {
    sprintf(buf, "%s", (*((const char*)elem)) == 1 ? "true" : "false");
}
static void indent(char *buf, size_t *pos, size_t n) {
    memset(buf + *pos, ' ', n);
    *pos += n;
}

static void build_string(char *buf, size_t *pos, char *base,
                         int dim, int dims, const int *shape,
                         const int *strides, size_t item_size,
                         fmt_func_t fmtfn, int depth) {
    buf[(*pos)++] = '[';

    if (dim < dims - 1) {
        buf[(*pos)++] = '\n';
        indent(buf, pos, (depth + 1) * 4);
    }

    for (size_t i = 0; i < shape[dim]; ++i) {
        if (i) {
            buf[(*pos)++] = ',';
            if (dim < dims - 1) {
                buf[(*pos)++] = '\n'; 
                indent(buf, pos, (depth + 1) * 4);
            }
            else { 
                buf[(*pos)++] = ' ';
            }
        }
        char *cur = base + i * strides[dim] * item_size;
        if (dim == dims - 1) {
            char tmp[16];
            fmtfn(tmp, sizeof tmp, cur);
            size_t len = strlen(tmp);
            memcpy(buf + *pos, tmp, len);
            *pos += len;
        } else {
            build_string(buf, pos, cur, dim + 1, dims, shape, strides,
                         item_size, fmtfn, depth + 1);
        }
    }

    if (dim < dims - 1) {
        buf[(*pos)++] = '\n';
        indent(buf, pos, depth * 4);
    }
    buf[(*pos)++] = ']';
}

char* array_to_string(void* p, size_t item_size, int* shape, int dims, fmt_func_t formatter) {
    int total = 1;
    for (int i = 0; i < dims; i++) {
        total *= shape[i];
    }
    
    int* strides = malloc(sizeof(int) * dims);
    if (!strides)
        return NULL;

    strides[dims - 1] = 1;
    for (int i = dims - 2; i >= 0; --i)
        strides[i] = strides[i + 1] * shape[i + 1];

    // Rough buffer size: 
    // Assume <=16 chars per element + 2 chars per bracket comma + 1 for '\0'
    size_t buf_size = (size_t)total * 16 + dims * 2 + 1;
    char* buf = malloc(buf_size);
    if (!buf) {
        free(strides);
        return NULL;
    }

    size_t pos = 0;
    build_string(buf, &pos, (char*)p, 0, dims, shape, strides, item_size, formatter, 0);
    buf[pos] = '\0';
    return buf;
}