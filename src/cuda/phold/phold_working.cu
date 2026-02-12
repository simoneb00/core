#include <cuda/cuda_gpu.h>
#include <cuda/queues.h>
#include <cuda/kernels.h>
#include <cuda/random.h>
#include "phold.h"

#define EVENT 1

typedef struct {
  curandState_t cr_state;
  uint64_t complete_events;
} PHoldState;

typedef struct {
  curandState_t *cr_state;
  uint64_t *complete_events;
} PHoldStateNodes;

#define DURATION 10

__device__ static uint population = 16000;
__device__ int32_t start_events = 1;
__device__ double p_remote = 0.25;
__device__ double phold_mean = 1000;
__device__ double phold_lookahead = 10000;
__device__ uint32_t num_lps = 256*1024;

extern uint events_per_node;
extern __device__ EQs	eq;
extern "C" uint get_n_nodes();
extern "C" uint get_n_lps();
extern "C" uint get_n_nodes_per_lp();
extern "C" uint get_n_blocks();

__device__ static PHoldStateNodes PHoldState_nodes;
curandState_t *rand_state_PHoldState;
uint64_t *complete_events;
uint *sim_bo;
uint *sim_so;
uint *sim_uo;
uint *sim_ql;
Event *sim_events;


char malloc_nodes(uint n_nodes) {
  cudaError_t err;
  PHoldStateNodes h_PHoldState_nodes;

  // allocate state for each LP
  rand_state_PHoldState = (curandState_t *)malloc(sizeof(curandState_t) * n_nodes);
  complete_events = (uint64_t *)malloc(sizeof(uint64_t) * n_nodes);

  if(!sim_bo) sim_bo = (uint*)malloc(sizeof(uint) * n_nodes);
  if(!sim_so) sim_so = (uint*)malloc(sizeof(uint) * n_nodes);
  if(!sim_uo) sim_uo = (uint*)malloc(sizeof(uint) * n_nodes);
  if(!sim_ql) sim_ql = (uint*)malloc(sizeof(uint) * n_nodes);

  // allocate messages for each LP
  if(!sim_events) sim_events = (Event*)malloc(sizeof(Event) * n_nodes * events_per_node);

  if (!rand_state_PHoldState || !complete_events || !sim_events) {printf("no memory for HOST side model state"); exit(1); }

  err = cudaMalloc(&(h_PHoldState_nodes.cr_state), sizeof(curandState_t) * n_nodes);
  if (err != cudaSuccess) { return 0; }
  err = cudaMalloc(&(h_PHoldState_nodes.complete_events), sizeof(uint64_t) * n_nodes);
  if (err != cudaSuccess) { return 0; }
  cudaMemcpyToSymbol(PHoldState_nodes, &h_PHoldState_nodes, sizeof(PHoldStateNodes));

  return 1;
}

void free_nodes() {
  PHoldStateNodes h_PHoldState_nodes;
  cudaMemcpyFromSymbol(&h_PHoldState_nodes, PHoldState_nodes, sizeof(PHoldStateNodes));
  cudaFree(h_PHoldState_nodes.cr_state);
  cudaFree(h_PHoldState_nodes.complete_events);
}

__device__
void set_model_params(int params[], uint n_params) {}

__device__
int get_lookahead() {
	return 0;
}

__device__
void collect_statistics(uint nid) {
	return;
}

__device__
void print_statistics() {
	printf("STATISTICS NOT AVAILABLE");
}

#if OPTM_SYNC == 1
__device__ // private
void reverse_event_type_1(Event *event) {}

__device__
void roll_back_event(Event *event) {}

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
char phold(uint64_t me, double now, Event *event)
{
  
  curandState_t *cr_state = &(PHoldState_nodes.cr_state[me]);

  PHoldState state = {
    .complete_events = PHoldState_nodes.complete_events[me],
  };
  state.complete_events++;
  busy_loop(DURATION);

  PHoldState old_state = {
    .cr_state = *cr_state,
    .complete_events = state.complete_events,
  };
  
  lp_id_t dest = me;
  if (curand(cr_state) <= p_remote) 
  {
    dest = ((lp_id_t)((curand(cr_state) * num_lps)));
  }


  struct PHoldMessage new_event = { 0 };
  Envelope e = {
    .priority = 5.0,
    .sender = me
  };
  Event new_msg = {
    .envelope = e,
    .payload = new_event
  };

  new_msg.receiver = dest;
  new_msg.timestamp = now + random_exp(cr_state, phold_mean) + phold_lookahead;
  new_msg.sender = me;
  new_msg.type = EVENT;
  char res = append_event_to_queue(&new_msg);
  if (res == 0) {
    PHoldState_nodes.complete_events[me] = old_state.complete_events;
    PHoldState_nodes.cr_state[me] = old_state.cr_state;
    return 11;
  }

  PHoldState_nodes.complete_events[me] = state.complete_events;

  return 1;
}

__device__
void init_node(uint nid) {

  lp_id_t me = nid;
  curand_init(nid, 0, 0, &(PHoldState_nodes.cr_state[nid]));
  curandState_t *cr_state = &(PHoldState_nodes.cr_state[me]);

  PHoldState state = {
    .complete_events = PHoldState_nodes.complete_events[me],
  };
  struct PHoldMessage new_event = { 0 };
  
  Envelope e = {
    .priority = 5.0,
    .sender = me
  };

  Event new_msg = {
    .envelope = e,
    .payload = new_event
  };

  for ( int32_t i = 0 ; i < start_events; i++ )
  {
    new_msg.receiver = me;
    new_msg.timestamp = random_exp(cr_state, phold_mean) + phold_lookahead;
    new_msg.sender = me;
    new_msg.type = EVENT;
    append_event_to_queue(&new_msg);

  }

  PHoldState_nodes.complete_events[me] = state.complete_events;
}

__device__
void reinit_node(uint nid, int gvt) {}

__device__
char handle_event(Event *message)
{
	uint type = message->type;

	if (type == 1) {
		return phold(message->receiver, message->timestamp, message);
	} else {
		return 0;
	}
}


extern "C" {
#include <core/core.h>
#include <lp/expose_lp_state.h>
extern void process_device_align_msg(unsigned lid, simtime_t time);

}

void copy_nodes_from_host(uint n_nodes) {}
void copy_nodes_to_host(uint n_nodes) {}

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
    rand_state_PHoldState[i] = *state;
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
	//printf("aligned memory from HOST to DEVICE");

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
    *state = rand_state_PHoldState[i];
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
