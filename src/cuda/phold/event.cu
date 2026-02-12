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


#include "event.h"


#ifdef COMPADS_DESL

__device__
char events_are_equal(Event *event_1, Event *event_2) {
    if (event_1->type      != event_2->type     ||
        event_1->sender    != event_2->sender   ||
        event_1->receiver  != event_2->receiver ||
        event_1->timestamp != event_2->timestamp) {
        return 0;
    }

    if (event_1->payload_size != event_2->payload_size) {
        return 0;
    }

    for (uint64_t i = 0; i < event_1->payload_size; i++) {
        if (event_1->payload[i] != event_2->payload[i]) {
            return 0;
        }
    }

    return 1;
}

#else

__device__
char events_are_equal(Event *event_1, Event *event_2) {
	if (
	event_1->type != event_2->type ||
	event_1->sender != event_2->sender ||
	event_1->receiver != event_2->receiver ||
	event_1->timestamp != event_2->timestamp) {
		return 0;
	}

	return 1;
}

#endif
