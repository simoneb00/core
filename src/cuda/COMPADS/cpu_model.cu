/**
 * @file test/tests/integration/phold.c
 *
 * @brief A simple and stripped phold implementation
 *
 * SPDX-FileCopyrightText: 2008-2023 HPDCS Group <rootsim@googlegroups.com>
 * SPDX-License-Identifier: GPL-3.0-only
 */
extern "C" {

#include <ROOT-Sim.h>
#include <ftl/ftl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "cpu_curand.h"
#include "settings.h"
}


#ifndef NUM_LPS
#define NUM_LPS (1024 * 512)
#endif

#ifndef NUM_THREADS
#define NUM_THREADS 0
#endif

#define EVENT 1

unsigned int mean = 10000;

#define DURATION 100

extern "C" {
#include <unistd.h>
#include "settings_cpu.h"
static simtime_t lookahead = 1000;

struct simulation_configuration conf;

struct PHoldState { 
   curandState_t cr_state; 
   uint64_t complete_events; 
};

typedef struct Envelope {
  lp_id_t sender;
  float priority;
} Envelope;

typedef struct PHoldMessage {
  uint64_t dummy;
} PHoldMessage;

typedef struct Message {
  Envelope envelope;
  uint64_t receiver;
  uint32_t type;
  double timestamp;
  PHoldMessage payload;
} Message;

int32_t cpu_start_events = 1;
double cpu_p_remote = 0.25;

double phold_mean = 10000.0;
double phold_lookahead = 1000.0;
uint32_t num_lps = 256 * 1024;

void __attribute__ ((noinline)) busy_loop(unsigned long long max) {
    for (unsigned long long i = 0; i < max; i++) {
        __asm__ volatile("pause" : "+g" (i) : :);
    }
}

static void phold(lp_id_t me, simtime_t now, void *msg, curandState_t *state)
{
  busy_loop(DURATION);
  
  lp_id_t dest = me;
  if (cpu_curand(state) <= cpu_p_remote) 
  {
    dest = ((lp_id_t)((cpu_curand(state) * num_lps)));
  }

  struct PHoldMessage new_event = { 0 };
  Envelope e = {
    .sender = me,
    .priority = 5.0
  };
  Message new_msg = {
    .envelope = e,
  };

  ScheduleNewEvent(dest, now + cpu_random_exp(state, mean) + lookahead, EVENT, &new_msg, sizeof(Message));
}

void ProcessEvent(lp_id_t me, simtime_t now, unsigned event_type, const void *content, unsigned size, void *state_ptr)
{
  switch(me) {
  case 0 ... 262143: {
    /* phold */
    switch(event_type) {
      case LP_INIT: {
        
        // INITIALIZING STATE
        curandState_t *state = (curandState_t *)state_ptr;
        state = (curandState_t *)rs_malloc(sizeof(curandState_t));
        if (state == NULL) abort();
        cpu_curand_init(me, 0, 0, state);
        SetState(state);

        struct PHoldMessage new_event = { 0 };
        
        Envelope e = {
          .sender = me,
          .priority = 5.0,
        };

        Message new_msg = {
          .envelope = e,
        };

        for ( int32_t i = 0 ; i < cpu_start_events; i++ )
        {
          ScheduleNewEvent(me, cpu_random_exp(state, mean) + lookahead, EVENT, &new_msg, sizeof(Message));
        }


        break;
      }
      case EVENT: {
        struct Message * msg = (struct Message *)content;
        phold(me, now, msg, (curandState_t *)state_ptr);
        break;
      }
      case LP_REINIT:
      case LP_FINI: 
        break;
    default:
      fprintf(stderr, "[ERROR]: EVENT TYPE %u UNKNOWN", event_type);
      puts("");
      abort();
    }
  break;
  }
  }
}

bool CanEnd(lp_id_t me, const void *snapshot){ 
	(void)me; (void)snapshot; return false; 
}
}

#define CPU  1
#define GPU  2
#define FTL  3


#define PERIODIC 0
#define MONITOR_LEADER 1
#define AIMD 2

int main(int argc, char *argv[])
{	
	int mode = 1;
	int ftlh = 0;
	if(argc >= 2){
		mode = atoi(argv[1]);
	}
	if(argc == 3){
		ftlh = atoi(argv[2]);
		if(ftlh > 2) { printf("unknown ftl_h\n");exit(1);}
	}
    conf.lps = NUM_LPS,
    conf.n_threads = NUM_THREADS,
    conf.termination_time = END_SIM_GVT,
    conf.gvt_period = GVT_PERIOD,
    conf.log_level = LOG_INFO,
    conf.stats_file = "phold",
    conf.ckpt_interval = 0,
    conf.core_binding = true,
    conf.serial = false,
    conf.use_gpu = mode & GPU,
    conf.use_cpu = mode & CPU,
    conf.dispatcher = ProcessEvent,
    conf.committed = CanEnd,
	conf.ftl_heuristic = ftlh_map[ftlh],
	RootsimInit(&conf);
	return RootsimRun();
}
