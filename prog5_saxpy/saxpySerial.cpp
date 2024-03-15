#include <immintrin.h>

void saxpySerial(int N,
                       float scale,
                       float X[],
                       float Y[],
                       float result[])
{
    __m128 s = _mm_set_ps1(scale);

    for (int i = 0; i < N; i += 4) {
        __m128 x = _mm_load_ps(X + i);
        __m128 y = _mm_load_ps(Y + i);
        __m128 z = _mm_mul_ps(s, x);
        z = _mm_add_ps(z, y);
        _mm_stream_ps(result + i, z);
    }

}

