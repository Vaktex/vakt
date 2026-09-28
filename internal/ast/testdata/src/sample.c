#include <stdio.h>
#include <string.h>

static int counter = 0;

/* Copies input. */
void copy(char *dst, const char *src) {
    strcpy(dst, src);
}

int main(int argc, char **argv) {
    char buf[16];
    copy(buf, argv[1]);
    return 0;
}
