/*  This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>. */


#ifndef State_h
#define State_h

#ifdef PHOLD_OLD
#include <curand_kernel.h>

typedef struct {
	curandState_t cr_state;
} State;
#endif

#ifdef PHOLD_DESL
#include <curand_kernel.h>

typedef struct {
  curandState_t cr_state;
} State;

#endif

#ifdef COMPADS_DESL

#define MAX_BUFFERS 16
#define MAX_BUFFER_SIZE 32

typedef __uint128_t test_rng_state;

typedef struct lp_buffer {
	unsigned count;
	int next;
	uint64_t data[MAX_BUFFER_SIZE];
} buffer;

typedef struct {
	unsigned events;
	unsigned buffer_count;
	uint32_t total_checksum;
	test_rng_state rng_state;
	buffer buffers[MAX_BUFFERS];
	int head;
  unsigned sent_antimsgs;
} State;
#endif


#endif
