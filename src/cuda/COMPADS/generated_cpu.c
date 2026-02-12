#include <ROOT-Sim.h>
#include <test.h>
#include <application.h>
#ifndef NUM_LPS
#define NUM_LPS 1
#endif

#ifndef NUM_THREADS
#define NUM_THREADS 0
#endif

#define MAX_BUFFERS 256
#define MAX_BUFFERS_SIZE 512
#define SEND_PROBABILITY 0.05
#define ALLOC_PROBABILITY 0.2
#define DEALLOC_PROBABILITY 0.2
#define DOUBLING_PROBABILITY 0.5
#define NULLING_PROBABILITY 0.3
#define COMPLETE_EVENTS 15000
#define N_LPS 256

typedef struct Envelope {
  lp_id_t sender;
  float priority;
} Envelope;

typedef struct Message {
  Envelope envelope;
  uint64_t receiver;
  uint32_t type;
  double timestamp;
  uint64_t payload[MAX_BUFFERS_SIZE];
  uint64_t payload_size;
} Message;

extern uint32_t const  model_expected_output[];


bool CanEnd(uint64_t me, void const *snapshot)
{
  struct lp_state *state = ((struct lp_state *)((snapshot)));
  return state->events >= COMPLETE_EVENTS;
}



void ProcessEvent(lp_id_t me, simtime_t now, unsigned event_type, const void *content, unsigned size, void *state_ptr)
{
  switch(me) {
  case 0 ... 255: {
    /* COMPADS */
    switch(event_type) {
      case LP_INIT: {
        struct lp_state * state = (struct lp_state *)state_ptr;
        // INITIALIZING STATE
        struct lp_state *state_to_init = (struct lp_state *)state;
        state_to_init = rs_malloc(sizeof(*state_to_init));
        if (state_to_init == NULL)
          abort();
        SetState(state_to_init);
        state = state_to_init;

        rng_init(&state->rng_state, (((test_rng_state)((me) + 1))) * 4390023366657240769ULL);
        
        uint32_t buffers_to_allocate = rng_random(&state->rng_state) * MAX_BUFFERS;
        for ( uint32_t i = 0 ; i < buffers_to_allocate; ++i )
        {
          uint32_t c = rng_random(&state->rng_state) * MAX_BUFFERS_SIZE / sizeof(uint64_t);
          state->head = allocate_buffer(state, NULL, c);
          state->buffer_count++;
        }

        
        
        ScheduleNewEvent(me, 20 * rng_random(&state->rng_state), LOOP, NULL, 0);

        break;
      }
      case LP_FINI: {
        struct lp_state * state = (struct lp_state *)state_ptr;
        struct Message * msg = (struct Message *)content;
        if (model_expected_output[me] != state->total_checksum) 
        {
          puts("[ERROR} Incorrect output!");
          abort();
        }


        break;
      }
      case LOOP: {
        struct lp_state * state = (struct lp_state *)state_ptr;
        struct Message * msg = (struct Message *)content;
        if (state->events >= COMPLETE_EVENTS) 
        {
          return;
        }

        
        if (rng_random(&state->rng_state) < NULLING_PROBABILITY) 
        {
          return;
        }

        state->events++;
        
        
        ScheduleNewEvent(me, now + rng_random(&state->rng_state) * 10, LOOP, NULL, 0);
        
        uint64_t dest = rng_random(&state->rng_state) * N_LPS;
        if (rng_random(&state->rng_state) < DOUBLING_PROBABILITY && dest != me) 
        {
          
          ScheduleNewEvent(dest, now + rng_random(&state->rng_state) * 10, LOOP, NULL, 0);
        }

        
        if (state->buffer_count != 0) 
        {
          state->total_checksum = read_buffer(state->head, rng_random(&state->rng_state) * state->buffer_count, state->total_checksum);
        }

        
        if (state->buffer_count < MAX_BUFFERS && rng_random(&state->rng_state) < ALLOC_PROBABILITY) 
        {
          uint32_t c = rng_random(&state->rng_state) * MAX_BUFFERS_SIZE / sizeof(uint64_t);
          state->head = allocate_buffer(state, NULL, c);
          state->buffer_count++;
        }

        
        if (state->buffer_count != 0 && rng_random(&state->rng_state) < DEALLOC_PROBABILITY) 
        {
          state->head = deallocate_buffer(state->head, rng_random(&state->rng_state) * state->buffer_count);
          state->buffer_count--;
        }

        if (state->buffer_count != 0 && rng_random(&state->rng_state) < SEND_PROBABILITY) 
        {
          uint32_t i = rng_random(&state->rng_state) * state->buffer_count;
          struct lp_buffer *to_send = get_buffer(state->head, i);
          uint64_t *data = to_send->data;
          
          dest = rng_random(&state->rng_state) * N_LPS;
          
          Envelope env = {
            .priority = 5.0,
            .sender = me
          };

          Message event3 = {
            .envelope = env,
          };
          for (int i = 0; i < to_send->count; i++) {
            event3.payload[i] = to_send->data[i];
          }
          event3.payload_size = to_send->count;

          ScheduleNewEvent(dest, now + rng_random(&state->rng_state) * 10, RECEIVE, &event3, sizeof(Message));
          
          state->head = deallocate_buffer(state->head, i);
          state->buffer_count--;
        }


        break;
      }
      case RECEIVE: {
        struct lp_state * state = (struct lp_state *)state_ptr;
        struct Message * msg = (struct Message *)content;
        if (state->events >= COMPLETE_EVENTS) 
        {
          return;
        }

        
        if (rng_random(&state->rng_state) < NULLING_PROBABILITY) 
        {
          return;
        }

        if (state->buffer_count >= MAX_BUFFERS) 
        {
          return;
        }

        state->head = allocate_buffer(state, ((uint32_t *)((msg->payload))), msg->payload_size);
        state->buffer_count++;

        break;
      }
    default:
      fprintf(stderr, "[ERROR]: EVENT TYPE %u UNKNOWN", event_type);
      puts("");
      abort();
    }
  break;
  }
  }
}

