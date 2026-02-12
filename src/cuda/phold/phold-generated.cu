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


#include <cuda/cuda_gpu.h>
#include <cuda/queues.h>
#include <cuda/kernels.h>
#include <cuda/random.h>

#include "settings.h"
#include "settings_gpu.h"


#define EVENT 100

typedef struct {
    curandState_t   *cr_state;
    uint64_t        *complete_events; 
} PHoldNodes;

// stato
typedef struct {
	curandState_t   cr_state;
    uint64_t        complete_events;
} phold_state;

__device__ static PHoldNodes nodes;
__device__ static uint	population;
__device__ static int lookahead;
__device__ static int	mean;


extern uint events_per_node;
extern __device__ EQs	eq;
extern "C" uint get_n_nodes();
extern "C" uint get_n_lps();
extern "C" uint get_n_nodes_per_lp();
extern "C" uint get_n_blocks();




__device__ int32_t start_events = 1;
__device__ double p_remote = 0.25;
__device__ uint32_t num_lps = 5000;


curandState_t *simulation_snapshot;
uint64_t *complete_events;
uint *sim_bo;
uint *sim_so;
uint *sim_uo;
uint *sim_ql;
Event *sim_events;


char malloc_nodes(uint n_nodes) {
	cudaError_t err;

	PHoldNodes h_nodes;
	
    // alloca stato per tutti gli LP
    simulation_snapshot = (curandState_t*) malloc(sizeof(curandState_t)*n_nodes);
    complete_events = (uint64_t *)malloc(sizeof(uint64_t)*n_nodes);

	if(!sim_bo) sim_bo = (uint*)malloc(sizeof(uint) * n_nodes);
	if(!sim_so) sim_so = (uint*)malloc(sizeof(uint) * n_nodes);
	if(!sim_uo) sim_uo = (uint*)malloc(sizeof(uint) * n_nodes);
	if(!sim_ql) sim_ql = (uint*)malloc(sizeof(uint) * n_nodes);

    // alloca i messaggi per tutti gli LP
	if(!sim_events) sim_events = (Event*)malloc(sizeof(Event) * n_nodes * events_per_node);

	if(!simulation_snapshot || !complete_events || !sim_events) {printf("no memory for HOST side model state\n"); exit(1); }

	err = cudaMalloc(&(h_nodes.cr_state), sizeof(curandState_t) * n_nodes);
	if (err != cudaSuccess) { return 0; }
    err = cudaMalloc(&(h_nodes.complete_events), sizeof(uint64_t) * n_nodes);
	if (err != cudaSuccess) { return 0; }
	cudaMemcpyToSymbol(nodes, &h_nodes, sizeof(PHoldNodes));

	return 1;
}

void free_nodes() {
	PHoldNodes h_nodes;
	cudaMemcpyFromSymbol(&h_nodes, nodes, sizeof(PHoldNodes));
	cudaFree(h_nodes.cr_state);
    cudaFree(h_nodes.complete_events);
}

__device__
void set_model_params(int params[], uint n_params) {
	population = params[0];
	lookahead = params[1];
	mean = params[2];
}

__device__
int get_lookahead() {
	return lookahead;
}


__device__
void init_node(uint nid) {
	curand_init(nid, 0, 0, &(nodes.cr_state[nid]));
    lp_id_t me = nid;

    struct PHoldMessage new_event = { 0 };
  
    Envelope e = {
        .priority = 5.0,
        .sender = me
    };

    Event new_msg{};
    new_msg.envelope = e;
    new_msg.payload  = new_event;

    for ( int32_t i = 0 ; i < start_events; i++ ) {
        new_msg.receiver = me;
        new_msg.timestamp = random_exp(&(nodes.cr_state[nid]), mean) + lookahead;
        new_msg.type = EVENT;
		new_msg.sender = me;
        append_event_to_queue(&new_msg);
    }
}

__device__
void reinit_node(uint nid, int gvt) {

	lp_id_t me = nid;

	curandState_t *cr_state = &(nodes.cr_state[nid]);

	struct PHoldMessage new_event = { 0 };

	Envelope e = {
        .priority = 5.0,
        .sender = me
    };

    Event new_msg{};
    new_msg.envelope = e;
    new_msg.payload  = new_event;

	for (uint i = 0; i < start_events; i++) {
		new_msg.receiver = me;
		new_msg.timestamp = random_exp(&(nodes.cr_state[nid]), mean) + lookahead;
        new_msg.type = EVENT;
		new_msg.sender = me;

		char res = append_event_to_queue(&new_msg);
	}
}

__device__
char phold(uint64_t me, double now, Event *msg)
{
  curandState_t *cr_state = &(nodes.cr_state[me]);
  nodes.complete_events[me]++;
  // s->complete_events++;
  //busy_loop(1000.0);
  
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
  Event new_msg;
  new_msg.envelope = e;
  new_msg.payload = new_event;
  

  new_msg.receiver = dest;
  new_msg.timestamp = now + random_exp(cr_state, mean) + lookahead;
  new_msg.type = EVENT;
  new_msg.sender = me;
  char res = append_event_to_queue(&new_msg);
  if (res == 0) {
	printf("append_event_to_queue returned 0");
	return 11;	
  }

  return 1;
}

__device__
char handle_event(Event *message)
{
  switch(message->receiver) {
    case 0: {
      /* phold */
      switch(message->type) {
        case EVENT: {
          return phold(message->receiver, message->timestamp, message);
        }
        case LP_FINI: {
          return 0;
        }
      default:
        printf("[ERROR]: EVENT TYPE %u UNKNOWN", message->type);
        //abort();
      }
    break;
    }
  }
}


__device__
void collect_statistics(uint nid) {
	return;
}

__device__
void print_statistics() {
	printf("STATISTICS NOT AVAILABLE\n");
}

#if OPTM_SYNC == 1
__device__ // private
void reverse_event_type_1(Event *event) {
	uint nid = event->receiver;
	uint lpid = nid / g_nodes_per_lp;

	State *old_state = delete_last_state(lpid);
	nodes.cr_state[nid] = old_state->cr_state;

	Event *antimsg = delete_last_antimsg(lpid);
	undo_event(antimsg);
}
#endif

#if OPTM_SYNC == 1
__device__
void roll_back_event(Event *event) {
	uint type = event->type;

	if (type == 1) {
		reverse_event_type_1(event);
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
static void busy_loop(double duration) {}



extern "C" {
#include <core/core.h>
#include <lp/expose_lp_state.h>
extern void process_device_align_msg(unsigned lid, simtime_t time);

}



void copy_nodes_from_host(uint n_nodes) {
	PHoldNodes h_nodes;
	cudaMemcpyFromSymbol(&h_nodes, nodes, sizeof(PHoldNodes));
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
	PHoldNodes h_nodes;
	cudaMemcpyFromSymbol(&h_nodes, nodes, sizeof(PHoldNodes));
	cudaMemcpy(simulation_snapshot, h_nodes.cr_state, sizeof(curandState_t) * n_nodes, cudaMemcpyDeviceToHost);

	EQs h_eq;
	cudaMemcpyFromSymbol(&h_eq, eq, sizeof(EQs));

	cudaMemcpy(sim_bo, h_eq.bo, sizeof(uint) * n_nodes, cudaMemcpyDeviceToHost);
	cudaMemcpy(sim_so, h_eq.so, sizeof(uint) * n_nodes, cudaMemcpyDeviceToHost);
	cudaMemcpy(sim_uo, h_eq.uo, sizeof(uint) * n_nodes, cudaMemcpyDeviceToHost);
	cudaMemcpy(sim_ql, h_eq.ql, sizeof(uint) * n_nodes, cudaMemcpyDeviceToHost);
	cudaMemcpy(sim_events, h_eq.events, sizeof(Event) * n_nodes * events_per_node, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	//printf("transferring mem frm DEV t HST\n");

}



extern "C" int pack_and_insert_gpu_event(unsigned des_node, unsigned sen_node, int ts, unsigned type){
	uint gpu_lp = des_node/get_n_nodes_per_lp();
	uint idx = __sync_fetch_and_add(sim_ql + gpu_lp, 1);

	if(idx >= get_n_nodes_per_lp()*events_per_node){
		printf("adding more events that queue capacity\n");
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

//	printf("A - copying events from SIM to HOST by %u from %u to %u \n", rid, start, i-1);
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

//	printf("B - copying events from SIM to HOST by %u from %u to %u : #events %u(%u)overall capacity %u\n", rid, start, i-1, cnt_c, cnt_a, events_per_node*get_n_nodes());

	transfer_per_thread_events(gvt);

}


extern "C" void align_device_to_host(unsigned threads_per_block){

	copy_nodes_from_host(global_config.lps);

	cudaDeviceSynchronize();
	//printf("aligned memory from HOST to DEVICE\n");

	kernel_sort_event_queues<<<get_n_blocks(), threads_per_block>>>();
	cudaDeviceSynchronize();
	//printf("sort queues \n");

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
	//printf("copying states from HOST to SIM by %u from %u to %u\n", rid, start, i-1);

	uint pushed_events= 0;
	if(rid < get_n_lps()){
		for(i=0;i<get_n_lps()/global_config.n_threads;i++){
			uint lp = rid * (get_n_lps()/global_config.n_threads) + i;
			uint zero_idx  = lp*get_n_nodes_per_lp()*events_per_node;
			uint base_idx  = sim_bo[lp];
			uint start_idx = sim_so[lp];
			uint end_idx   = sim_uo[lp];
			//if(rid == 0) printf("base %u start %u end %u size %u\n", base_idx, start_idx, end_idx, get_n_nodes_per_lp()*events_per_node);
			while(start_idx != end_idx){
				uint effective = (base_idx+start_idx) % (get_n_nodes_per_lp()*events_per_node);
				Event *cur = sim_events+zero_idx+effective;
				//printf("A scheduling for %u a message from %u at %u\n", cur->receiver, cur->sender, cur->timestamp);
				custom_schedule_from_gpu(gvt, cur->sender, cur->receiver, (simtime_t) cur->timestamp, cur->type, NULL, 0);
				start_idx++;
				pushed_events++;
			}
		}
		//printf("copying events from HOST to SIM by %u from %u to %u GPU #LPS %u -- events pushed %u\n",
		//	rid, rid * (get_n_lps()/global_config.n_threads),rid * (get_n_lps()/global_config.n_threads)+get_n_lps()/global_config.n_threads-1, get_n_lps(), pushed_events);
	}

}
    
