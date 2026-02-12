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

#include <stdint.h>
#include "state.h"

#pragma once

typedef struct {
    double          priority;
    uint            sender;
} Envelope; 

#if defined(PHOLD_OLD)
typedef struct {
	uint	type;
	uint	sender;
	uint	receiver;
	int	    timestamp;
} Event;

#elif defined(PHOLD_DESL)

struct PHoldMessage {
    long dummy_data;
};

typedef struct {
    Envelope                envelope;
	uint	                receiver;
    uint                    sender;
    uint	                type;
	int	                    timestamp;
    struct PHoldMessage     payload;
} Event;

#elif defined(COMPADS_DESL)

typedef struct Event {
    Envelope envelope;
    uint64_t sender;
    uint64_t receiver;
    uint32_t type;
    double timestamp;
    uint64_t payload[MAX_BUFFER_SIZE];
    uint64_t payload_size;
} Event;

#else

    #error "No PHOLD/COMPADS model selected"

#endif

__device__
char events_are_equal(Event *event_1, Event *event_2);
