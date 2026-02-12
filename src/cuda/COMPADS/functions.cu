/**
 * @file test/tests/integration/correctness/functions.c
 *
 * @brief Helper functions of the model used to verify the runtime correctness
 *
 * SPDX-FileCopyrightText: 2008-2025 HPDCS Group <rootsim@googlegroups.com>
 * SPDX-License-Identifier: GPL-3.0-only
 */
#include "application.h"

#include <string.h>

__device__
uint32_t crc_update(const uint64_t *buf, size_t n, uint32_t crc);

__device__
buffer *get_buffer(State *state, unsigned i)
{
	int cur = state->head;
	while (i-- && cur != -1) {
		cur = state->buffers[cur].next;
	}
	return (cur == -1) ? NULL : &state->buffers[cur];
}

__device__
uint32_t read_buffer(State *state, unsigned i, uint32_t old_crc, unsigned me)
{

	//if (!me) printf("Reading buffer with index %u, old_crc is %u\n", i, old_crc);

    buffer *b = get_buffer(state, i);
    if (!b) {
        //if (!me) printf("Buffer not found\n");
		return old_crc;
	}

	//if (!me) printf("Found buffer with %u elements\n", b->count);
	uint32_t result = crc_update(b->data, b->count, old_crc); 
	//if (!me) printf("Result is %u\n", result);

    return result;
}


__device__
buffer *allocate_buffer(State *state, const unsigned *data, unsigned count, unsigned me)
{
	int idx = state->buffer_count;

    buffer *b = &state->buffers[idx];
    b->next = state->head;
	b->count = count;

    if (data) {
        for (unsigned i = 0; i < count; i++)
            b->data[i] = data[i];
    } else {
		for (unsigned i = 0; i < count; i++)
            b->data[i] = rng_random_u(&state->rng_state);
	}

	state->head = idx;

	//if (!me) {
	//	printf("Allocated new buffer with %u elements: ", count);
	//	for (unsigned i = 0; i < count; i++) {
	//		printf("%u, ", b->data[i]);
	//	}
	//	printf("\n");
	//}

    return b;
}


__device__
void deallocate_buffer(State *state, unsigned i)
{
	/*
	buffer *prev = NULL;
	buffer *to_free = head;

	for(unsigned j = 0; j < i; j++) {
		prev = to_free;
		to_free = to_free->next;
	}

	if(prev != NULL) {
		prev->next = to_free->next;
		rs_free(to_free);
		return head;
	}

	prev = head->next;
	rs_free(head);
	return prev;
*/

	int prev = -1;
    int cur = state->head;

    while (i-- && cur != -1) {
        prev = cur;
        cur = state->buffers[cur].next;
    }

    if (cur == -1)
        return;

    if (prev != -1)
        state->buffers[prev].next = state->buffers[cur].next;
    else
        state->head = state->buffers[cur].next;

    int last = state->buffer_count - 1;
    if (cur != last) {
        state->buffers[cur] = state->buffers[last];

        if (state->head == last)
            state->head = cur;

        for (unsigned j = 0; j < state->buffer_count; j++) {
            if (state->buffers[j].next == last)
                state->buffers[j].next = cur;
        }
    }

}

__device__ __constant__
static uint32_t crc_table[256];


void crc_table_init(void)
{

	uint32_t host_crc_table[256];

	uint32_t n = 256;
	while(n--) {
		uint32_t c = n;
		int k = 8;
		while(k--) {
			if(c & 1) {
				c = 0xedb88320UL ^ (c >> 1);
			} else {
				c = c >> 1;
			}
		}
		host_crc_table[n] = c;
	}

	cudaError_t err = cudaMemcpyToSymbol(crc_table, host_crc_table, sizeof(host_crc_table));
    if (err != cudaSuccess) {
        printf("cudaMemcpyToSymbol failed: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    printf("CRC table initialized and copied to GPU\n");
}

__device__
uint32_t crc_update(const uint64_t *buf, size_t n, uint32_t crc)
{
	uint32_t c = crc ^ 0xffffffffUL;
	while(n--) {
		unsigned k = 64;
		do {
			k -= 8;
			c = crc_table[(c ^ (buf[n] >> k)) & 0xff] ^ (c >> 8);
		} while(k);
	}
	return c ^ 0xffffffffUL;
}