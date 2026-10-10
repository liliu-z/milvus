#include <cstdio>
extern "C" void goZapLogExt(int severity, const char*, int, int,
                            const char* msg, int msg_len) {
    if (severity > 0) {
        std::fwrite(msg, 1, msg_len, stderr);
        std::fputc('\n', stderr);
    }
}
