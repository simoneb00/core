#include <cuda/cuda_gpu.h>
#include <cuda/queues.h>
#include <cuda/kernels.h>
#include <cuda/random.h>
#include "phold.h"
#include "settings_gpu.h"

#define EVENT 1

#define COMPLETE_EVENTS 5000
#define BUSY_LOOP_DURATION 100
#define NUM_LPS (256 * 1024)

__device__ static Nodes nodes;


__device__ static uint population;
__device__ static int	phold_lookahead = 1000.0;
__device__ static int	phold_mean = 10000.0;

__device__ simtime_t p_remote = 0.25;

__device__ int32_t start_events = 1;

extern uint events_per_node;
extern __device__ EQs	eq;
extern "C" uint get_n_nodes();
extern "C" uint get_n_lps();
extern "C" uint get_n_nodes_per_lp();
extern "C" uint get_n_blocks();


curandState_t *simulation_snapshot;
uint *sim_bo;
uint *sim_so;
uint *sim_uo;
uint *sim_ql;
Event *sim_events;

 
char malloc_nodes(uint n_nodes) {
  cudaError_t err;
  Nodes h_nodes;

  // allocate state for each LP
  simulation_snapshot = (curandState_t *)malloc(sizeof(curandState_t) * n_nodes);

  if(!sim_bo) sim_bo = (uint*)malloc(sizeof(uint) * n_nodes);
  if(!sim_so) sim_so = (uint*)malloc(sizeof(uint) * n_nodes);
  if(!sim_uo) sim_uo = (uint*)malloc(sizeof(uint) * n_nodes);
  if(!sim_ql) sim_ql = (uint*)malloc(sizeof(uint) * n_nodes);

  // allocate messages for each LP
  if(!sim_events) sim_events = (Event*)malloc(sizeof(Event) * n_nodes * events_per_node);

  if (!simulation_snapshot || !sim_events) {printf("no memory for HOST side model state"); exit(1); }

  err = cudaMalloc(&(h_nodes.cr_state), sizeof(curandState_t) * n_nodes);
  if (err != cudaSuccess) { return 0; }
  cudaMemcpyToSymbol(nodes, &h_nodes, sizeof(nodes));

  return 1;
}

void free_nodes() {
  Nodes h_nodes;
  cudaMemcpyFromSymbol(&h_nodes, nodes, sizeof(nodes));
  cudaFree(h_nodes.cr_state);
}

__device__
void set_model_params(int params[], uint n_params) {
}

__device__
int get_lookahead() {
	return phold_lookahead;
}

__device__
void collect_statistics(uint nid) {
	return;
}

__device__
void print_statistics() {
	printf("STATISTICS NOT AVAILABLE");
}

__device__ static int hot_phase_count = 0;

__device__
static uint get_receiver(uint me, curandState_t *cr_state, int now)
{
	int cur_hot_phase = (now / PHASE_WINDOW_SIZE);
	double HOT_FRACTION = load_trace[cur_hot_phase];
	return random(cr_state, HOT_FRACTION * g_n_nodes)/(HOT_FRACTION);
}

__device__
char PHoldClass_behavior(uint64_t me, double now, Event *msg)
{

#if OPTM_SYNC == 1
	uint lpid = me / g_nodes_per_lp;

	if (state_queue_is_full(lpid)) { return 12; }
	if (antimsg_queue_is_full(lpid)) { return 13; }
#endif

  curandState_t *cr_state = &(nodes.cr_state[me]);

  /* save state before updates */
  State old_state = {
    .cr_state = nodes.cr_state[me],
  };

  busy_loop(BUSY_LOOP_DURATION);
  
  //lp_id_t dest = me;
  //if (curand(cr_state) <= p_remote) 
  //{
  //  dest = ((lp_id_t)((curand(cr_state)) * NUM_LPS));
  //}

  lp_id_t dest = get_receiver(me, cr_state, msg->timestamp);

  struct PHoldMessage new_event = { 0 };
  
  Envelope env_mnxqv_g0a0 = {
    .priority = 5.0,
    .sender = me
  };

  Event new_msg_mnxqv_g0a0 = {
    .envelope = env_mnxqv_g0a0,
    .payload = new_event
  };

  new_msg_mnxqv_g0a0.receiver = dest;
  new_msg_mnxqv_g0a0.timestamp = now + random_exp(cr_state, phold_mean) + phold_lookahead;
  new_msg_mnxqv_g0a0.sender = me;
  new_msg_mnxqv_g0a0.type = EVENT;
  char append_res = append_event_to_queue(&new_msg_mnxqv_g0a0);
  if (append_res == 0) {
    nodes.cr_state[me] = old_state.cr_state;
    return 11;
  }

#if OPTM_SYNC == 1
  append_state_to_queue(&old_state, me / g_nodes_per_lp);
  append_antimsg_to_queue(&new_msg_mnxqv_g0a0);
#endif


  return 1;
}

__device__
void init_node(uint nid) {
  lp_id_t me = nid;
  curand_init(nid, 0, 0, &(nodes.cr_state[nid]));
  curandState_t *cr_state = &(nodes.cr_state[me]);

  struct PHoldMessage new_event = { 0 };
  for ( int32_t i = 0 ; i < start_events; i++ )
  {
    {
      Envelope env_mnxqv_a0b0a0 = {
        .priority = 5.0,
        .sender = me
      };

      Event new_msg_mnxqv_a0b0a0 = {
        .envelope = env_mnxqv_a0b0a0,
        .payload = new_event
      };

      new_msg_mnxqv_a0b0a0.receiver = me;
      new_msg_mnxqv_a0b0a0.timestamp = random_exp(cr_state, phold_mean) + phold_lookahead;
      new_msg_mnxqv_a0b0a0.sender = me;
      new_msg_mnxqv_a0b0a0.type = EVENT;
      char append_res = append_event_to_queue(&new_msg_mnxqv_a0b0a0);

    }

  }
}

__device__
void reinit_node(uint nid, int gvt) {
  lp_id_t me = nid;
  curandState_t *cr_state = &(nodes.cr_state[me]);

  struct PHoldMessage new_event = { 0 };
  for ( int32_t i = 0 ; i < start_events; i++ )
  {
    {
      Envelope env_mnxqv_a0b0a0 = {
        .priority = 5.0,
        .sender = me
      };

      Event new_msg_mnxqv_a0b0a0 = {
        .envelope = env_mnxqv_a0b0a0,
        .payload = new_event
      };

      new_msg_mnxqv_a0b0a0.receiver = me;
      new_msg_mnxqv_a0b0a0.timestamp = gvt + random_exp(cr_state, phold_mean) + phold_lookahead;
      new_msg_mnxqv_a0b0a0.sender = me;
      new_msg_mnxqv_a0b0a0.type = EVENT;
      char append_res = append_event_to_queue(&new_msg_mnxqv_a0b0a0);

    }

  }
}

__device__
char handle_event(Event *message)
{
  if (message->receiver >= 0 && message->receiver <= 262143) {
    /* PHoldClass_behavior */
    switch(message->type) {
      case EVENT: {
        return PHoldClass_behavior(message->receiver, message->timestamp, message);
        break;
      }
    default:
      printf("[ERROR]: EVENT TYPE %u UNKNOWN", message->type);
    }
  }
}

#if OPTM_SYNC == 1
__device__ // private
void PHoldClass_behavior_reverse(Event *event) {

  uint nid = event->receiver;
	uint lpid = nid / g_nodes_per_lp;

	State *old_state = delete_last_state(lpid);
	nodes.cr_state[nid] = old_state->cr_state;

	Event *antimsg = delete_last_antimsg(lpid);
	undo_event(antimsg);
}

__device__
void roll_back_event(Event *message) {
  if (message->receiver >= 0 && message->receiver <= 262143) {
    /* PHoldClass_behavior */
    switch(message->type) {
      case EVENT: {
        return PHoldClass_behavior_reverse(message);
        break;
      }
    default:
      printf("[ERROR]: EVENT TYPE %u UNKNOWN", message->type);
    }
  }}

__device__
uint get_number_states(Event *event) {
	return 1;
}

__device__
uint get_number_antimsgs(Event *event) {
	return 1;
}
#endif

extern "C" {
#include <core/core.h>
#include <lp/expose_lp_state.h>
extern void process_device_align_msg(unsigned lid, simtime_t time);

}

void copy_nodes_from_host(uint n_nodes) {
  Nodes h_nodes;
	cudaMemcpyFromSymbol(&h_nodes, nodes, sizeof(Nodes));
	cudaMemcpy(h_nodes.cr_state, simulation_snapshot, sizeof(curandState_t) * n_nodes, cudaMemcpyHostToDevice);

	EQs h_eq;
	cudaMemcpyFromSymbol(&h_eq, eq, sizeof(EQs));
	cudaMemcpy(h_eq.bo,sim_bo, sizeof(uint) * n_nodes, cudaMemcpyHostToDevice);
	cudaMemcpy(h_eq.so,sim_so, sizeof(uint) * n_nodes, cudaMemcpyHostToDevice);
	cudaMemcpy(h_eq.uo,sim_uo, sizeof(uint) * n_nodes, cudaMemcpyHostToDevice);
	cudaMemcpy(h_eq.ql,sim_ql, sizeof(uint) * n_nodes, cudaMemcpyHostToDevice);
	cudaMemcpy(h_eq.events,  sim_events, sizeof(Event) * n_nodes * events_per_node, cudaMemcpyHostToDevice);
}

void copy_nodes_to_host(uint n_nodes) {
  Nodes h_nodes;
	cudaMemcpyFromSymbol(&h_nodes, nodes, sizeof(Nodes));
	cudaMemcpy(simulation_snapshot, h_nodes.cr_state, sizeof(curandState_t) * n_nodes, cudaMemcpyDeviceToHost);

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
    simulation_snapshot[i] = *state;
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
    *state = simulation_snapshot[i];
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
