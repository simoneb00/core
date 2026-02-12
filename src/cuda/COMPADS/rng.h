#pragma once
#include <stdint.h>

typedef __uint128_t test_rng_state;

#ifdef __CUDACC__
#define HD __host__ __device__ __forceinline__
#else
#define HD inline
#endif

HD void rng_init(test_rng_state *rng_state, test_rng_state initseq)
{
    *rng_state = (initseq << 1u) | 1u;
}


HD uint64_t rng_random_u(test_rng_state *rng_state)
{
    const __uint128_t multiplier =
        (((__uint128_t)0x0fc94e3bf4e9ab32ULL) << 64)
        + 0x866458cd56f5e605ULL;

    *rng_state *= multiplier;
    return (uint64_t)(*rng_state >> 64u);
}


HD double rng_random(test_rng_state *rng_state)
{
    uint64_t u_val = rng_random_u(rng_state);
    double ret = 0.0;

    if (u_val != 0) {
#ifdef __CUDA_ARCH__
        unsigned lzs = __clzll(u_val) + 1;
#else
        unsigned lzs = __builtin_clzll(u_val) + 1;
#endif
        u_val <<= lzs;
        u_val >>= 12;

        uint64_t exp = (uint64_t)(1023 - lzs) << 52;
        u_val |= exp;

#ifdef __CUDA_ARCH__
        ret = __longlong_as_double((long long)u_val);
#else
        memcpy(&ret, &u_val, sizeof(double));
#endif
    }

    return ret;
}
