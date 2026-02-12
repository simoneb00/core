#if defined(PHOLD_DESL) || defined(PHOLD_OLD)

#define NUM_THREADS 12
#define NUM_LPS (256*1024)
#define ENABLE_HOT 1
#define PHASE_WINDOW_SIZE (8*1000*1000)
#define HOT_PHASE_PERIOD 2
#define END_SIM_GVT  (64*1000*1000)

#else

#define NUM_THREADS 12
#define NUM_LPS (256)
#define ENABLE_HOT 1
#define PHASE_WINDOW_SIZE (8*1000*1000)
#define HOT_PHASE_PERIOD 2
#define END_SIM_GVT  (64*1000*1000)

#endif
