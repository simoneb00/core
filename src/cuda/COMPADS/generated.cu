#include <cuda/cuda_gpu.h>
#include <cuda/queues.h>
#include <cuda/kernels.h>
#include <cuda/random.h>
#include "application.h"

#define LOOP 0
#define RECEIVE 1


__device__ Nodes nodes;
__device__ static int lookahead = 1;

 //simulation parameters

__device__ extern const uint32_t  model_expected_output[];

extern uint events_per_node;
extern __device__ EQs	eq;
extern "C" uint get_n_nodes();
extern "C" uint get_n_lps();
extern "C" uint get_n_nodes_per_lp();
extern "C" uint get_n_blocks();

// one pointer variable for each member of the State struct
unsigned *events_snapshot;
unsigned *buffer_count_snapshot;
uint32_t *total_checksum_snapshot;
test_rng_state *rng_state_snapshot;
buffer *buffers_snapshot;
int *head_snapshot;

uint *sim_bo;
uint *sim_so;
uint *sim_uo;
uint *sim_ql;
Event *sim_events;

__device__ unsigned lp_to_check = 1000;

char malloc_nodes(uint n_nodes)
{
    cudaError_t err;

    Nodes h_nodes;
    events_snapshot = (unsigned *)malloc(sizeof(unsigned)*n_nodes); 
    buffer_count_snapshot = (unsigned *)malloc(sizeof(unsigned)*n_nodes); 
    total_checksum_snapshot = (uint32_t *)malloc(sizeof(uint32_t)*n_nodes); 
    rng_state_snapshot = (test_rng_state *)malloc(sizeof(test_rng_state)*n_nodes); 
    buffers_snapshot = (buffer *)malloc(sizeof(buffer)*MAX_BUFFERS*n_nodes); 
    head_snapshot = (int *)malloc(sizeof(int)*n_nodes); 
    
    if(!sim_bo) sim_bo = (uint*)malloc(sizeof(uint) * n_nodes);
    if(!sim_so) sim_so = (uint*)malloc(sizeof(uint) * n_nodes);
    if(!sim_uo) sim_uo = (uint*)malloc(sizeof(uint) * n_nodes);
    if(!sim_ql) sim_ql = (uint*)malloc(sizeof(uint) * n_nodes);

    if (!sim_events) sim_events = (Event *)malloc(sizeof(Event) * n_nodes * events_per_node);
    
    if (!events_snapshot || !buffer_count_snapshot || !total_checksum_snapshot || !rng_state_snapshot || !buffers_snapshot || !head_snapshot) {
      printf("no memory for HOST side model state\n"); 
      exit(1);
    }

    err = cudaMalloc(&(h_nodes.events), sizeof(unsigned) * n_nodes);
    if (err == cudaSuccess) err = cudaMalloc(&(h_nodes.buffer_count), sizeof(unsigned) * n_nodes);
    if (err == cudaSuccess) err = cudaMalloc(&(h_nodes.total_checksum), sizeof(uint32_t) * n_nodes);
    if (err == cudaSuccess) err = cudaMalloc(&(h_nodes.rng_state), sizeof(test_rng_state) * n_nodes);
    if (err == cudaSuccess) err = cudaMalloc(&(h_nodes.buffers), sizeof(buffer) * MAX_BUFFERS * n_nodes);
    if (err == cudaSuccess) err = cudaMalloc(&(h_nodes.head), sizeof(int) * n_nodes);

    if (err != cudaSuccess) { printf("cudaMalloc lp_state failed\n"); return 0; }

    cudaMemset(h_nodes.events,         0, sizeof(unsigned) * n_nodes);
    cudaMemset(h_nodes.buffer_count,   0, sizeof(unsigned) * n_nodes);
    cudaMemset(h_nodes.total_checksum, 0, sizeof(uint32_t) * n_nodes);
    cudaMemset(h_nodes.rng_state,      0, sizeof(test_rng_state) * n_nodes);
    cudaMemset(h_nodes.buffers,        0, sizeof(buffer) * MAX_BUFFERS * n_nodes);
    cudaMemset(h_nodes.head,           0, sizeof(int) * n_nodes);

    cudaMemcpyToSymbol(nodes, &h_nodes, sizeof(Nodes));
    if (err != cudaSuccess) {
        printf("cudaMemcpyToSymbol failed: %s\n", cudaGetErrorString(err));
        return 0;
    }

    return 1;
}

void free_nodes() {
  Nodes h_nodes;
  cudaMemcpyFromSymbol(&h_nodes, nodes, sizeof(Nodes));
  cudaFree(h_nodes.events);
  cudaFree(h_nodes.buffer_count);
  cudaFree(h_nodes.total_checksum);
  cudaFree(h_nodes.rng_state);
  cudaFree(h_nodes.buffers);
  cudaFree(h_nodes.head);
}

__device__
int get_lookahead() {
	return 1;
}

__device__ void print128(test_rng_state val) {
  uint64_t high = (uint64_t)(val >> 64);
  uint64_t low  = (uint64_t)val;
  if (high > 0) printf("0x%llx%016llx", high, low);
  else printf("0x%llx", low);
}


__device__ void print_state(lp_id_t me, State *s) {
   
    printf("\n========= COMPARISON [LP: %lu] =========\n", me);
    
    // Tabella comparativa per i campi scalari
    printf("%-15s | %-15s | %-15s\n", "Field", "Local State", "Global Nodes");
    printf("------------------------------------------------------------\n");
    printf("%-15s | %-15u | %-15u\n", "Events", s->events, nodes.events[me]);
    printf("%-15s | %-15u | %-15u\n", "Buf Count", s->buffer_count, nodes.buffer_count[me]);
    printf("%-15s | %-15u | %-15u\n", "Checksum", s->total_checksum, nodes.total_checksum[me]);
    printf("%-15s | %-15d | %-15d\n", "Head", s->head, nodes.head[me]);
    
    printf("%-15s | ", "RNG State");
    print128(s->rng_state); printf(" | "); print128(nodes.rng_state[me]);
    printf("\n");

    // Verifica integrità primo Buffer (se esistente)
    if (s->buffer_count > 0) {
        printf("------------------------------------------------------------\n");
        // Nel calcolo globale l'offset è me * MAX_BUFFERS
        buffer *global_buf = &(nodes.buffers[me * MAX_BUFFERS]); 
        printf("First Buffer Check:\n");
        printf("  Local  -> count: %u, data[0]: %llu\n", s->buffers[0].count, s->buffers[0].data[0]);
        printf("  Global -> count: %u, data[0]: %llu\n", global_buf->count, global_buf->data[0]);
    }
    printf("============================================================\n");
}

__device__
void init_node(uint nid)
{
  lp_id_t me = nid;

  State state;
  state.events = nodes.events[me];
  state.buffer_count = nodes.buffer_count[me];
  state.total_checksum = nodes.total_checksum[me];
  state.rng_state = nodes.rng_state[me];
  for(int i = 0; i < MAX_BUFFERS; i++) {
    state.buffers[i] = nodes.buffers[me * MAX_BUFFERS + i];
  }
  state.head = nodes.head[me];

  rng_init(&state.rng_state, ((test_rng_state)(me + 1)) * 4390023366657240769ULL);

  for (unsigned i = 0; i < MAX_BUFFERS; i++) {
      state.buffers[i].count = 0;
      state.buffers[i].next = -1;
  }

  uint32_t buffers_to_allocate = rng_random(&state.rng_state) * MAX_BUFFERS;
  for (uint32_t i = 0; i < buffers_to_allocate; ++i) {
      uint32_t c = rng_random(&state.rng_state) * MAX_BUFFER_SIZE / sizeof(uint64_t);
      allocate_buffer(&state, NULL, c, me);
      state.buffer_count++;
  }

  Event event;
  event.receiver = me;
  event.sender   = me;
  event.type     = LOOP;
  event.timestamp = 20 * rng_random(&state.rng_state) + (double)lookahead;
  
  append_event_to_queue(&event);

  // save state
  nodes.events[me] = state.events;
  nodes.buffer_count[me] = state.buffer_count;
  nodes.total_checksum[me] = state.total_checksum;
  nodes.rng_state[me] = state.rng_state;
  for(int i = 0; i < MAX_BUFFERS; i++) {
    nodes.buffers[me * MAX_BUFFERS + i] = state.buffers[i];
  }
  nodes.head[me] = state.head;

  if (event.receiver == lp_to_check) { 
    print_state(me, &state);
    printf("[%ld] Scheduled initial LOOP event at timestamp %.6f, rng_state is ", me, event.timestamp);
    print128(nodes.rng_state[me]);
    printf("\n");
  }
}

__device__
void reinit_node(uint nid, int gvt) {
  Event event;
  event.receiver = nid;
  event.sender   = nid;
  event.type     = LOOP;
  event.timestamp = 20 * rng_random(&nodes.rng_state[nid]) + (double)lookahead;
  
  if (nid / g_nodes_per_lp == 18) printf("[%u] Rescheduled initial LOOP event\n", nid);

  append_event_to_queue(&event);
}

__device__
char handle_event_loop(Event *message) {

  lp_id_t me = message->receiver;
  double now = message->timestamp;

  if (me == lp_to_check) {
    printf("[%ld] Handling event: Sender: %-5llu | Receiver: %-5llu | Type: %u | TS: %-10.6f\n", 
      me,
      (unsigned long long)message->sender, 
      (unsigned long long)message->receiver, 
      message->type, 
      message->timestamp);
  }


  #if OPTM_SYNC == 1
    uint lpid = me / g_nodes_per_lp;

    if (state_queue_is_full(lpid)) { return 12; }
    if (antimsg_queue_is_full(lpid)) { return 13; }
  #endif

  // get state

  State state;
  state.events = nodes.events[me];
  state.buffer_count = nodes.buffer_count[me];
  state.total_checksum = nodes.total_checksum[me];
  state.rng_state = nodes.rng_state[me];
  for(int i = 0; i < MAX_BUFFERS; i++) {
      state.buffers[i] = nodes.buffers[me * MAX_BUFFERS + i];
    }
  state.head = nodes.head[me];
  state.sent_antimsgs = 0;

  // create a snapshot of the buffers associated to 
  buffer buffers_snapshot[MAX_BUFFERS]; 
  for(int i = 0; i < MAX_BUFFERS; i++) {
      buffers_snapshot[i] = state.buffers[i]; 
  }

  // create state snapshot
  State old_state;
  old_state.events = nodes.events[me];
  old_state.buffer_count = nodes.buffer_count[me];
  old_state.total_checksum = nodes.total_checksum[me];
  old_state.rng_state = nodes.rng_state[me];
  for(int i = 0; i < MAX_BUFFERS; i++) {
    old_state.buffers[i] = buffers_snapshot[i];
  }
  old_state.head = nodes.head[me];
  old_state.sent_antimsgs = 0;


  //if (!me) print_state(me, &state);

  if (rng_random(&state.rng_state) < NULLING_PROBABILITY) {

    // save state
    nodes.events[me] = state.events;
    nodes.buffer_count[me] = state.buffer_count;
    nodes.total_checksum[me] = state.total_checksum;
    nodes.rng_state[me] = state.rng_state;
    for(int i = 0; i < MAX_BUFFERS; i++) {
      nodes.buffers[me * MAX_BUFFERS + i] = state.buffers[i];
    }
    nodes.head[me] = state.head;

    if (me == lp_to_check) {
      printf("[LOOP, %ld] NULLING_PROBABILITY, checksum is %-15u, rng_state is ", me, nodes.total_checksum[me]);
      print128(nodes.rng_state[me]);
      printf("\n");
    }

    #if OPTM_SYNC == 1
      append_state_to_queue(&old_state, lpid);
    #endif

    return 1;
  }

  state.events++;
  //if (me == 0) printf("[%ld] Processed %u events\n", me, state.events);


  /* self-loop */
  Event event1;
  event1.receiver = me;
  event1.sender   = me;
  event1.type     = LOOP;
  event1.timestamp = now + rng_random(&state.rng_state) * 10 + (double)lookahead;

  //if (event1.receiver == 0) printf("[%ld] Scheduling event to 0 at time %f\n", me, event1.timestamp);

  //if (event1.receiver == 153) printf("[SCHEDULE] LP %lu, type %u, timestamp %.6f\n", me, event1.type, event1.timestamp);

  if (!append_event_to_queue(&event1)) {
      printf("[%ld] append_event_to_queue failed\n", me);

      // restore state (only buffers, other fields have not been committed)
      for(int i = 0; i < MAX_BUFFERS; i++) {
        nodes.buffers[me * MAX_BUFFERS + i] = buffers_snapshot[i];
      }
      return 11;
  }

  #if OPTM_SYNC == 1
    append_antimsg_to_queue(&event1);
  #endif

  if (me == lp_to_check) { 
    printf("[%ld] Appended antimsg to queue (SELF LOOP): Sender: %-5llu | Receiver: %-5llu | Type: %u | TS: %-10.6f\n", 
      me,
      (unsigned long long)event1.sender, 
      (unsigned long long)event1.receiver, 
      event1.type, 
      event1.timestamp);
  }
  old_state.sent_antimsgs++;

  /* possible doubling */
  uint64_t dest = rng_random(&state.rng_state) * N_LPS;
  if (rng_random(&state.rng_state) < DOUBLING_PROBABILITY && dest != me) {

      Event event2;
      event2.receiver = dest;
      event2.sender   = me;
      event2.type     = LOOP;
      event2.timestamp = now + rng_random(&state.rng_state) * 10 + (double)lookahead;

      //if (event2.receiver == 153) printf("[SCHEDULE] LP %lu, type %u, timestamp %.6f\n", me, event2.type, event2.timestamp);

      if (!append_event_to_queue(&event2)) {
          printf("[%ld] append_event_to_queue failed\n", me);

          // restore state (only buffers, other fields have not been committed)
          for(int i = 0; i < MAX_BUFFERS; i++) {
            nodes.buffers[me * MAX_BUFFERS + i] = buffers_snapshot[i];
          }

          return 11;
      }

      #if OPTM_SYNC == 1
        append_antimsg_to_queue(&event2);
      #endif

      if (me == lp_to_check) { 
        printf("[%ld] Appended antimsg to queue (DOUBLING): Sender: %-5llu | Receiver: %-5llu | Type: %u | TS: %-10.6f\n", 
          me,
          (unsigned long long)event2.sender, 
          (unsigned long long)event2.receiver, 
          event2.type, 
          event2.timestamp);
      }
      old_state.sent_antimsgs++;
  }

  /* read buffer */
  if (state.buffer_count) {

      //if (!me) printf("\n\nUpdating checksum\n");
      uint32_t i = rng_random(&state.rng_state) * state.buffer_count;
      state.total_checksum = read_buffer(&state, i, state.total_checksum, me);
      //if (!me) printf("Updated checksum to value %u\n\n\n", state.total_checksum);

  }

  /* allocate */
  if (state.buffer_count < MAX_BUFFERS && rng_random(&state.rng_state) < ALLOC_PROBABILITY) {
      uint32_t c = rng_random(&state.rng_state) * MAX_BUFFER_SIZE / sizeof(uint64_t);
      allocate_buffer(&state, NULL, c, me);
      state.buffer_count++;
  }

  /* deallocate */
  if (state.buffer_count && rng_random(&state.rng_state) < DEALLOC_PROBABILITY) {
      uint32_t i = rng_random(&state.rng_state) * state.buffer_count;
      deallocate_buffer(&state, i);
      state.buffer_count--;
  }

  /* send RECEIVE event */

  if (state.buffer_count && rng_random(&state.rng_state) < SEND_PROBABILITY) {
      uint32_t i = rng_random(&state.rng_state) * state.buffer_count;
      buffer *to_send = get_buffer(&state, i);

      Event event3;
      event3.receiver = rng_random(&state.rng_state) * N_LPS;
      event3.sender   = me;
      event3.type     = RECEIVE;
      event3.timestamp = now + rng_random(&state.rng_state) * 10 + (double)lookahead;

      event3.payload_size = to_send->count;
      for (unsigned j = 0; j < to_send->count; j++)
          event3.payload[j] = to_send->data[j];

      //if (event3.receiver == 153) printf("[SCHEDULE] LP %lu, type %u, timestamp %.6f\n", me, event3.type, event3.timestamp);

      if (!append_event_to_queue(&event3)) {      
          printf("[%ld] append_event_to_queue failed\n", me);
        
          // restore state (only buffers, other fields have not been committed)
          for(int i = 0; i < MAX_BUFFERS; i++) {
            nodes.buffers[me * MAX_BUFFERS + i] = buffers_snapshot[i];
          }

          return 11;
      }

      #if OPTM_SYNC == 1
        append_antimsg_to_queue(&event3);
      #endif

      if (me == lp_to_check) { 
        printf("[%ld] Appended antimsg to queue (SEND RECEIVE EVENT): Sender: %-5llu | Receiver: %-5llu | Type: %u | TS: %-10.6f\n", 
          me,
          (unsigned long long)event3.sender, 
          (unsigned long long)event3.receiver, 
          event3.type, 
          event3.timestamp);
      }

      old_state.sent_antimsgs++;

      deallocate_buffer(&state, i);
      state.buffer_count--;
  }

  // save state
  nodes.events[me] = state.events;
  nodes.buffer_count[me] = state.buffer_count;
  nodes.total_checksum[me] = state.total_checksum;
  nodes.rng_state[me] = state.rng_state;
  for(int i = 0; i < MAX_BUFFERS; i++) {
    nodes.buffers[me * MAX_BUFFERS + i] = state.buffers[i];
  }
  nodes.head[me] = state.head;

  if (me == lp_to_check) {
    printf("[LOOP, %ld], checksum is %-15u, rng_state is ", me, nodes.total_checksum[me]);
    print128(nodes.rng_state[me]);
    printf("\n");
  }

  //if (me == 149) {
  //  printf("[%ld] Appended %u antimsgs to queue\n", me, old_state.sent_antimsgs);
  //}

  #if OPTM_SYNC == 1
    append_state_to_queue(&old_state, lpid);
  #endif

  return 1;
}

__device__
char handle_event_receive(Event *message) {

  lp_id_t me = message->receiver;

  if (me == lp_to_check) {
    printf("[%ld] Handling event: Sender: %-5llu | Receiver: %-5llu | Type: %u | TS: %-10.6f\n", 
      me,
      (unsigned long long)message->sender, 
      (unsigned long long)message->receiver, 
      message->type, 
      message->timestamp);
  }


  if (me == lp_to_check) {
    printf("[RECEIVE] LP %ld received buffer with %u elements, content is: ", me, message->payload_size);
    for (unsigned i = 0; i < message->payload_size; i++) {
      printf("%u, ", message->payload[i]);
    }
    printf("\n");
  }


  #if OPTM_SYNC == 1
    uint lpid = me / g_nodes_per_lp;

    if (state_queue_is_full(lpid)) { return 12; }
    if (antimsg_queue_is_full(lpid)) { return 13; }
  #endif

  // get state
  State state;
  state.events = nodes.events[me];
  state.buffer_count = nodes.buffer_count[me];
  state.total_checksum = nodes.total_checksum[me];
  state.rng_state = nodes.rng_state[me];
  for(int i = 0; i < MAX_BUFFERS; i++) {
      state.buffers[i] = nodes.buffers[me * MAX_BUFFERS + i];
    }
  state.head = nodes.head[me];

  // create a snapshot of the buffers associated to 
  buffer buffers_snapshot[MAX_BUFFERS]; 
  for(int i = 0; i < MAX_BUFFERS; i++) {
      buffers_snapshot[i] = state.buffers[i]; 
  }

  // create state snapshot
  State old_state;
  old_state.events = nodes.events[me];
  old_state.buffer_count = nodes.buffer_count[me];
  old_state.total_checksum = nodes.total_checksum[me];
  old_state.rng_state = nodes.rng_state[me];
  for(int i = 0; i < MAX_BUFFERS; i++) {
    old_state.buffers[i] = buffers_snapshot[i];
  }
  old_state.head = nodes.head[me];
  old_state.sent_antimsgs = 0;

  #if OPTM_SYNC == 1
    append_state_to_queue(&old_state, lpid);
  #endif

  if (rng_random(&state.rng_state) < NULLING_PROBABILITY) {
      
    // save state
    nodes.events[me] = state.events;
    nodes.buffer_count[me] = state.buffer_count;
    nodes.total_checksum[me] = state.total_checksum;
    nodes.rng_state[me] = state.rng_state;
    for(int i = 0; i < MAX_BUFFERS; i++) {
      nodes.buffers[me * MAX_BUFFERS + i] = state.buffers[i];
    }
    nodes.head[me] = state.head;

    if (me == lp_to_check) {
      printf("[RECEIVE, %ld] NULLING_PROBABILITY, checksum is %-15u, rng_state is ", me, nodes.total_checksum[me]);
      print128(nodes.rng_state[me]);
      printf("\n");
    }
    
    return 1;
  }

  if (state.buffer_count >= MAX_BUFFERS) {

    // save state
    nodes.events[me] = state.events;
    nodes.buffer_count[me] = state.buffer_count;
    nodes.total_checksum[me] = state.total_checksum;
    nodes.rng_state[me] = state.rng_state;
    for(int i = 0; i < MAX_BUFFERS; i++) {
      nodes.buffers[me * MAX_BUFFERS + i] = state.buffers[i];
    }
    nodes.head[me] = state.head;
    
    if (me == lp_to_check) {
      printf("[RECEIVE, %ld] MAX_BUFFERS, checksum is %-15u, rng_state is ", me, nodes.total_checksum[me]);
      print128(nodes.rng_state[me]);
      printf("\n");
    }
    return 1;
  }

  allocate_buffer(&state, (uint32_t *)message->payload, message->payload_size, me);
  state.buffer_count++;
  
  // save state
  nodes.events[me] = state.events;
  nodes.buffer_count[me] = state.buffer_count;
  nodes.total_checksum[me] = state.total_checksum;
  nodes.rng_state[me] = state.rng_state;
  for(int i = 0; i < MAX_BUFFERS; i++) {
    nodes.buffers[me * MAX_BUFFERS + i] = state.buffers[i];
  }
  nodes.head[me] = state.head;

  if (me == lp_to_check) {
    printf("[RECEIVE, %ld], checksum is %-15u, rng_state is ", me, nodes.total_checksum[me]);
    print128(nodes.rng_state[me]);
    printf("\n");
  }
  return 1;
}

__device__
char handle_event(Event *message)
{
  uint type = message->type;

  //if (message->receiver == 0) printf("[0] Received event with type %u at time %f\n", type, message->timestamp);

  // check result
  if (nodes.events[message->receiver] == COMPLETE_EVENTS) {
    //printf("[%ld] Processed %d events, checking expected output...\n", message->receiver, nodes.events[message->receiver]);
    printf("[%ld] %-15u\n", message->receiver, nodes.total_checksum[message->receiver]);
    //if(model_expected_output[message->receiver] != nodes.total_checksum[message->receiver]) {
		//		printf("[ERROR] Incorrect output (expecting %u but found %-15u)!\n", model_expected_output[message->receiver], nodes.total_checksum[message->receiver]);
		//		asm("trap;");
		//	}
    //nodes.events[message->receiver]++;
    //return 1;
  } //else if (nodes.events[message->receiver] > COMPLETE_EVENTS) {return 1;}

  if (type == 0) {
    return handle_event_loop(message);
  } else if (type == 1) {
    return handle_event_receive(message);
  } else {
    printf("Unknown event type!\n");
    return 0;
  }
}


#if OPTM_SYNC == 1


  __device__
  void reverse_event_loop(Event *event) {
    uint me = event->receiver;
    uint lpid = me / g_nodes_per_lp;

    State *old_state = delete_last_state(lpid);
    
    // restore state 
    if (old_state != NULL) {

      nodes.events[me]         = old_state->events;
      nodes.buffer_count[me]   = old_state->buffer_count;
      nodes.total_checksum[me] = old_state->total_checksum;
      nodes.rng_state[me]      = old_state->rng_state;
      nodes.head[me]           = old_state->head;

      for(int i = 0; i < MAX_BUFFERS; i++) {
        nodes.buffers[me * MAX_BUFFERS + i] = old_state->buffers[i];
      }

      //if (me == 149) {
      //  printf("[%u] Removing %u antimsgs from queue\n", me, old_state->sent_antimsgs);
      //} 

      // undo sent events
      while (old_state->sent_antimsgs-- > 0) {
        Event *antimsg = delete_last_antimsg(lpid);

        if (me == lp_to_check) {
          printf("[%u] Undoing event: Sender: %-5llu | Receiver: %-5llu | Type: %u | TS: %-10.6f\n", 
            me,
            (unsigned long long)antimsg->sender, 
            (unsigned long long)antimsg->receiver, 
            antimsg->type, 
            antimsg->timestamp);
        }

        undo_event(antimsg, me);
      }

    } else {
      printf("[roll_back_event] old_state is NULL!\n");
      asm("trap;");
    }
  }

  __device__
  void reverse_event_receive(Event *event) {
    uint me = event->receiver;
    uint lpid = me / g_nodes_per_lp;

    State *old_state = delete_last_state(lpid);
    
    // restore state 
    if (old_state != NULL) {

      nodes.events[me]         = old_state->events;
      nodes.buffer_count[me]   = old_state->buffer_count;
      nodes.total_checksum[me] = old_state->total_checksum;
      nodes.rng_state[me]      = old_state->rng_state;
      nodes.head[me]           = old_state->head;

      for(int i = 0; i < MAX_BUFFERS; i++) {
        nodes.buffers[me * MAX_BUFFERS + i] = old_state->buffers[i];
      }
    } else {
      printf("[roll_back_event] old_state is NULL!\n");
    }
  }


  __device__
  void roll_back_event(Event *event) {
    uint type = event->type;

    if (event->receiver == lp_to_check) {      
      printf("[%u] Rollbacking event: Sender: %-5llu | Receiver: %-5llu | Type: %u | TS: %-10.6f\n", 
        event->receiver,
        (unsigned long long)event->sender, 
        (unsigned long long)event->receiver, 
        event->type, 
        event->timestamp);
        
    }

    if (type == 0) {
      reverse_event_loop(event);
    } else if (type == 1) {
      reverse_event_receive(event);
    }
  }

  __device__
  uint get_number_states(Event *event) {
    return 1;
  }

  __device__
  uint get_number_antimsgs(Event *event) {
    return 1;
  }

#endif

__device__
void collect_statistics(uint nid) {
	return;
}

__device__
void print_statistics() {
	printf("STATISTICS NOT AVAILABLE");
}

__device__
void set_model_params(int params[], uint n_params) {}



extern "C" {
#include <core/core.h>
#include <lp/expose_lp_state.h>
extern void process_device_align_msg(unsigned lid, simtime_t time);

}

/*
typedef struct {
  unsigned *events;
	unsigned *buffer_count;
	uint32_t *total_checksum;
	test_rng_state *rng_state;
	buffer *buffers;
	int *head;
} Nodes;


err = cudaMalloc(&(h_nodes.events), sizeof(unsigned) * n_nodes);
if (err == cudaSuccess) err = cudaMalloc(&(h_nodes.buffer_count), sizeof(unsigned) * n_nodes);
if (err == cudaSuccess) err = cudaMalloc(&(h_nodes.total_checksum), sizeof(uint32_t) * n_nodes);
if (err == cudaSuccess) err = cudaMalloc(&(h_nodes.rng_state), sizeof(test_rng_state) * n_nodes);
if (err == cudaSuccess) err = cudaMalloc(&(h_nodes.buffers), sizeof(buffer) * MAX_BUFFERS * n_nodes);
if (err == cudaSuccess) err = cudaMalloc(&(h_nodes.head), sizeof(int) * n_nodes);
*/

void copy_nodes_from_host(uint n_nodes) {
  Nodes h_nodes;
  cudaMemcpyFromSymbol(&h_nodes, nodes, sizeof(Nodes));
  cudaMemcpy(h_nodes.events, events_snapshot, sizeof(unsigned) * n_nodes, cudaMemcpyHostToDevice);
  cudaMemcpy(h_nodes.buffer_count, buffer_count_snapshot, sizeof(unsigned) * n_nodes, cudaMemcpyHostToDevice);
  cudaMemcpy(h_nodes.total_checksum, total_checksum_snapshot, sizeof(uint32_t) * n_nodes, cudaMemcpyHostToDevice);
  cudaMemcpy(h_nodes.rng_state, rng_state_snapshot, sizeof(test_rng_state) * n_nodes, cudaMemcpyHostToDevice);
  cudaMemcpy(h_nodes.buffers, buffers_snapshot, sizeof(buffer) * MAX_BUFFERS * n_nodes, cudaMemcpyHostToDevice);
  cudaMemcpy(h_nodes.head, head_snapshot, sizeof(int) * n_nodes, cudaMemcpyHostToDevice);

  EQs h_eq;
  cudaMemcpyFromSymbol(&h_eq, eq, sizeof(EQs));
  cudaMemcpy(h_eq.bo, sim_bo, sizeof(uint) * n_nodes, cudaMemcpyHostToDevice);
  cudaMemcpy(h_eq.so, sim_so, sizeof(uint) * n_nodes, cudaMemcpyHostToDevice);
  cudaMemcpy(h_eq.uo, sim_uo, sizeof(uint) * n_nodes, cudaMemcpyHostToDevice);
  cudaMemcpy(h_eq.ql, sim_ql, sizeof(uint) * n_nodes, cudaMemcpyHostToDevice);
  cudaMemcpy(h_eq.events, sim_events, sizeof(Event) * n_nodes * events_per_node, cudaMemcpyHostToDevice);
}



void copy_nodes_to_host(uint n_nodes) {  
  Nodes h_nodes;
  cudaMemcpyFromSymbol(&h_nodes, nodes, sizeof(Nodes));
  cudaMemcpy(events_snapshot, h_nodes.events, sizeof(unsigned) * n_nodes, cudaMemcpyDeviceToHost);
  cudaMemcpy(buffer_count_snapshot, h_nodes.buffer_count, sizeof(unsigned) * n_nodes, cudaMemcpyDeviceToHost);
  cudaMemcpy(total_checksum_snapshot, h_nodes.total_checksum, sizeof(uint32_t) * n_nodes, cudaMemcpyDeviceToHost);
  cudaMemcpy(rng_state_snapshot, h_nodes.rng_state, sizeof(test_rng_state) * n_nodes, cudaMemcpyDeviceToHost);
  cudaMemcpy(buffers_snapshot, h_nodes.buffers, sizeof(buffer) * MAX_BUFFERS * n_nodes, cudaMemcpyDeviceToHost);
  cudaMemcpy(head_snapshot, h_nodes.head, sizeof(int) * n_nodes, cudaMemcpyDeviceToHost);

  EQs h_eq;
  cudaMemcpyFromSymbol(&h_eq, eq, sizeof(EQs));

  cudaMemcpy(sim_bo, h_eq.bo, sizeof(uint) * n_nodes, cudaMemcpyDeviceToHost);
  cudaMemcpy(sim_so, h_eq.so, sizeof(uint) * n_nodes, cudaMemcpyDeviceToHost);
  cudaMemcpy(sim_uo, h_eq.uo, sizeof(uint) * n_nodes, cudaMemcpyDeviceToHost);
  cudaMemcpy(sim_ql, h_eq.ql, sizeof(uint) * n_nodes, cudaMemcpyDeviceToHost);
  cudaMemcpy(sim_events, h_eq.events, sizeof(Event) * n_nodes * events_per_node, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
}


extern "C" int pack_and_insert_gpu_event(unsigned des_node, unsigned sen_node, int ts, unsigned type){
	uint gpu_lp = des_node/get_n_nodes_per_lp();
	uint idx = __sync_fetch_and_add(sim_ql + gpu_lp, 1);

	if(idx >= get_n_nodes_per_lp()*events_per_node){
		printf("adding more events that queue capacity");
		exit(1);
	}

	uint base = get_n_nodes_per_lp()*events_per_node*gpu_lp;

	Event *tgt = sim_events+base+idx;
	tgt->receiver = des_node;
	tgt->sender   = sen_node;
	tgt->timestamp = ts;
	tgt->type = type;
	return 1;
}

extern "C" void align_device_to_host_parallel_states(unsigned rid, simtime_t gvt){
  unsigned start;
  unsigned i;

  start = global_config.lps+1;
  for(i=0;i<global_config.lps;i++){
    if(lid_to_rid(i) != rid && start == (global_config.lps+1)) continue;
    if(lid_to_rid(i) != rid && start != (global_config.lps+1)) break;
    if(start == (global_config.lps+1)) start = i;
    align_lp_state_to_gvt(gvt,i);
    curandState_t *state = (curandState_t*) get_lp_state_base_pointer(i);
    //rand_state_lp_state[i] = *state;
  }

  clean_per_thread_queue();

  if(!rid){
    bzero(sim_events, sizeof(Event) * get_n_nodes() * events_per_node);
    bzero(sim_bo, sizeof(uint) * get_n_nodes());
    bzero(sim_so, sizeof(uint) * get_n_nodes());
    bzero(sim_uo, sizeof(uint) * get_n_nodes());
    bzero(sim_ql, sizeof(uint) * get_n_nodes());
  }
}

extern "C" void align_device_to_host_parallel_events(unsigned rid, simtime_t gvt){
	unsigned cnt_a = 0;
	unsigned cnt_c = 0;
	unsigned start = (global_config.lps+1);
	unsigned i;

	start = global_config.lps+1;
	cnt_a = 0;
	cnt_c = 0;
	for(i=0;i<global_config.lps;i++){
		if(lid_to_rid(i) != rid && start == (global_config.lps+1)) continue;
		if(lid_to_rid(i) != rid && start != (global_config.lps+1)) break;
		if(start == (global_config.lps+1)) start = i;

		cnt_a += estimate_transfer_per_lp_events_without_filter(i);
		cnt_c += transfer_per_lp_events(i,gvt);
	}

//	printf("B - copying events from SIM to HOST by %u from %u to %u : #events %u(%u)overall capacity %u", rid, start, i-1, cnt_c, cnt_a, events_per_node*get_n_nodes());

	transfer_per_thread_events(gvt);

}

extern "C" void align_device_to_host(unsigned threads_per_block){

	copy_nodes_from_host(global_config.lps);

	cudaDeviceSynchronize();
	printf("aligned memory from HOST to DEVICE");

	kernel_sort_event_queues<<<get_n_blocks(), threads_per_block>>>();
	cudaDeviceSynchronize();
	//printf("sort queues ");

}

extern "C" void align_host_to_device(){
	copy_nodes_to_host(global_config.lps);
	cudaDeviceSynchronize();
}

extern "C" void align_host_to_device_parallel(simtime_t gvt){
  unsigned start = (global_config.lps+1);
  unsigned i;
  for(i=0;i<global_config.lps;i++){
    if(lid_to_rid(i) != rid && start == (global_config.lps+1)) continue;
    if(lid_to_rid(i) != rid && start != (global_config.lps+1)) break;
    if(start == (global_config.lps+1)) start = i;
    curandState_t *state = (curandState_t*) get_lp_state_base_pointer(i);
    //*state = rand_state_lp_state[i];
    process_device_align_msg(i, gvt);
  }

uint pushed_events= 0;
	if(rid < get_n_lps()){
		for(i=0;i<get_n_lps()/global_config.n_threads;i++){
			uint lp = rid * (get_n_lps()/global_config.n_threads) + i;
			uint zero_idx  = lp*get_n_nodes_per_lp()*events_per_node;
			uint base_idx  = sim_bo[lp];
			uint start_idx = sim_so[lp];
			uint end_idx   = sim_uo[lp];
			//if(rid == 0) printf("base %u start %u end %u size %u", base_idx, start_idx, end_idx, get_n_nodes_per_lp()*events_per_node);
			while(start_idx != end_idx){
				uint effective = (base_idx+start_idx) % (get_n_nodes_per_lp()*events_per_node);
				Event *cur = sim_events+zero_idx+effective;
				//printf("A scheduling for %u a message from %u at %u", cur->receiver, cur->sender, cur->timestamp);
				custom_schedule_from_gpu(gvt, cur->sender, cur->receiver, (simtime_t) cur->timestamp, cur->type, NULL, 0);
				start_idx++;
				pushed_events++;
			}
		}
		//printf("copying events from HOST to SIM by %u from %u to %u GPU #LPS %u -- events pushed %u",
		//	rid, rid * (get_n_lps()/global_config.n_threads),rid * (get_n_lps()/global_config.n_threads)+get_n_lps()/global_config.n_threads-1, get_n_lps(), pushed_events);
	}
}
