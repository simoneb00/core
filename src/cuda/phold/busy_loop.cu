#include "phold.h"

extern "C" __global__
void busy_loop_wrapper(unsigned long long cycles)
{
    busy_loop(cycles);
}