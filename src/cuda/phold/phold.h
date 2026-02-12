#pragma once

__device__ __noinline__
void busy_loop(unsigned long long cycles)
{
    unsigned long long start = clock64();
    while (clock64() - start < cycles) {
    }
}