#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>

__m128 _mm_fabs_ps(__m128 x) {
    __m128 zero = _mm_set_ps1(0.f);

    __m128 minusX = _mm_sub_ps(zero, x);
    __m128 xPositive = _mm_cmpge_ps(x, zero);
    return _mm_blendv_ps(minusX, x, xPositive);
}


void sqrtSerial(int N,
                float initialGuess,
                float values[],
                float output[])
{

    static const float kThreshold = 0.00001f;

    for (int i = 0; i < N; i += 4) {
        __m128 x = _mm_load_ps(values + i);
        __m128 guess = _mm_set_ps1(initialGuess);
        __m128 zero = _mm_set_ps1(0.f);
        __m128 half = _mm_set_ps1(.5f);
        __m128 one = _mm_set_ps1(1.f);
        __m128 three = _mm_set_ps1(3.f);
        __m128 threshold = _mm_set_ps1(kThreshold);

        __m128 error = _mm_mul_ps(guess, guess);
        error = _mm_mul_ps(error, x);
        error = _mm_sub_ps(error, one);
        error = _mm_fabs_ps(error);

        while (1) {
            __m128 comp = _mm_cmpgt_ps(error, threshold);
            if (_mm_test_all_zeros(_mm_castps_si128(comp), _mm_castps_si128(one))) break;

            __m128 first = _mm_mul_ps(guess, three);
            __m128 second = _mm_mul_ps(x, guess);
            second = _mm_mul_ps(second, guess);
            second = _mm_mul_ps(second, guess);
            first = _mm_sub_ps(first, second);
            first = _mm_mul_ps(first, half);
            guess = _mm_blendv_ps(guess, first, comp);

            first = _mm_mul_ps(guess, guess);
            first = _mm_mul_ps(first, x);
            first = _mm_sub_ps(first, one);
            first = _mm_fabs_ps(first);
            error = _mm_blendv_ps(error, first, comp);
        }

        __m128 out = _mm_mul_ps(x, guess);
        _mm_store_ps(output + i, out);
    }
}

