#ifndef _CUTS_H_
#define _CUTS_H_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "cuda_runtime.h"
#include "cuda.h"

#define datacost(pix,lab)     (datacost[(pix)*num_Labels+(lab)] )
#define smoothnesscost(lab1,lab2) (smoothnesscost[(lab1)+(lab2)*num_Labels] )

class GraphCuts {
public:
	// Ensures a cuda device is present
	// Initializes the grid, blocksize, etc.
	int graphCutsInit(int, int, int);

	int checkDevice();

	// Initializes memory
	void h_mem_init();

	// Allocates and initializes memory on device
	void d_mem_init();

	// Sets up the data values
	int graphCutsSetupDataTerm();

	// Sets up the smoothness values
	int graphCutsSetupSmoothTerm();

	// Sets up the Cue values
	int graphCutsSetupHCue();
	int graphCutsSetupVCue();

	// Sets up the graph
	int graphCutsSetupGraph();

	// Optimize Algorithm
	int graphCutsAtomicOptimize();
	int graphCutsStochasticOptimize();

	// push, pull, relabel
	void graphCutsStochastic();
	void graphCutsAtomic();

	// BFS
	void bfsLabeling();

	// Labels each pixel
	int graphCutsGetResult();

	// Frees all memory
	void graphCutsFreeMem();


	// Global Variables

	int *d_left_weight, *d_right_weight, *d_down_weight, *d_up_weight, *d_push_reser, *d_sink_weight;
	int *s_left_weight, *s_right_weight, *s_down_weight, *s_up_weight, *s_push_reser, *s_sink_weight;
	int *d_pull_left, *d_pull_right, *d_pull_down, *d_pull_up;

	int *d_stochastic, *d_stochastic_pixel, *d_terminate;

	int *dataTerm, *smoothTerm, *hCue, *vCue;
	int *dDataTerm, *dSmoothTerm, *dHcue, *dVcue, *dPixelLabel;

	int  *d_relabel_mask, *d_graph_heightr, *d_graph_heightw;

	int graph_size, size_int, width, height, graph_size1, width1, height1, depth, num_Labels;
	int blocks_x, blocks_y, threads_x, threads_y, num_of_blocks, num_of_threads_per_block;

	int *pixelLabel;

	bool *d_pixel_mask, h_over, *d_over, *h_pixel_mask;
	int *d_counter, *h_graph_height;
	int *h_reset_mem;
	int cueValues, deviceCheck, deviceCount;

	int *h_stochastic, *h_stochastic_pixel, *h_relabel_mask;
	int counter;
};


// Kernel Functions

__global__ void
kernel_push1_atomic(int *g_left_weight, int *g_right_weight, int *g_down_weight, int *g_up_weight,
int *g_sink_weight, int *g_push_reser, int *g_pull_left, int *g_pull_right, int *g_pull_down,
int *g_pull_up, int *g_relabel_mask, int *g_graph_height, int *g_height_write,
int graph_size, int width, int rows, int graph_size1, int width1, int rows1);

__global__ void
kernel_relabel_atomic(int *g_left_weight, int *g_right_weight, int *g_down_weight, int *g_up_weight,
int *g_sink_weight, int *g_push_reser, int *g_pull_left, int *g_pull_right, int *g_pull_down,
int *g_pull_up, int *g_relabel_mask, int *g_graph_height, int *g_height_write,
int graph_size, int width, int rows, int graph_size1, int width1, int rows1);

__global__ void
kernel_relabel_stochastic(int *g_left_weight, int *g_right_weight, int *g_down_weight, int *g_up_weight,
int *g_sink_weight, int *g_push_reser,
int *g_relabel_mask, int *g_graph_height, int *g_height_write,
int graph_size, int width, int rows, int graph_size1, int width1, int rows1, int *d_stochastic, int *g_block_num);

__global__ void
kernel_push2_atomic(int *g_left_weight, int *g_right_weight, int *g_down_weight, int *g_up_weight,
int *g_sink_weight, int *g_push_reser, int *g_pull_left, int *g_pull_right, int *g_pull_down, int *g_pull_up,
int *g_relabel_mask, int *g_graph_height, int *g_height_write,
int graph_size, int width, int rows, int graph_size1, int width1, int rows1);

__global__ void
kernel_End(int *g_stochastic, int *g_count_blocks, int *g_counter);


__global__ void
kernel_push1_start_atomic(int *g_left_weight, int *g_right_weight, int *g_down_weight, int *g_up_weight,
int *g_sink_weight, int *g_push_reser,
int *g_relabel_mask, int *g_graph_height, int *g_height_write,
int graph_size, int width, int rows, int graph_size1, int width1, int rows1, int *d_relabel, int *d_stochastic, int *d_counter, bool *d_finish);

__global__ void
kernel_push1_stochastic(int *g_left_weight, int *g_right_weight, int *g_down_weight, int *g_up_weight,
int *g_sink_weight, int *g_push_reser,
int *g_relabel_mask, int *g_graph_height, int *g_height_write,
int graph_size, int width, int rows, int graph_size1, int width1, int rows1, int *d_stochastic, int *g_block_num);

__global__ void
kernel_push2_stochastic(int *g_left_weight, int *g_right_weight, int *g_down_weight, int *g_up_weight,
int *g_sink_weight, int *g_push_reser, int *g_pull_left, int *g_pull_right, int *g_pull_down, int *g_pull_up,
int *g_relabel_mask, int *g_graph_height, int *g_height_write,
int graph_size, int width, int rows, int graph_size1, int width1, int rows1, int *d_relabel, int *d_stochastic, int *d_counter, bool *d_finish, int *g_block_num);

__global__ void
kernel_bfs_t(int *g_push_reser, int  *g_sink_weight, int *g_graph_height, bool *g_pixel_mask,
int vertex_num, int width, int height, int vertex_num1, int width1, int height1);

__global__ void
kernel_push_stochastic1(int *g_push_reser, int *s_push_reser, int *g_count_blocks, bool *g_finish, int *g_block_num, int width1);

__global__ void
kernel_push_atomic2(int *g_terminate, int *g_push_reser, int *s_push_reser, int *g_block_num, int width1);


__global__ void
kernel_push_stochastic2(int *g_terminate, int *g_relabel_mask, int *g_push_reser, int *s_push_reser, int *d_stochastic, int *g_block_num, int width1);

__global__ void
kernel_push1_start_stochastic(int *g_left_weight, int *g_right_weight, int *g_down_weight, int *g_up_weight,
int *g_sink_weight, int *g_push_reser,
int *g_relabel_mask, int *g_graph_height, int *g_height_write,
int graph_size, int width, int rows, int graph_size1, int width1, int rows1, int *d_relabel, int *d_stochastic, int *d_counter, bool *d_finish);


__global__ void
kernel_bfs(int *g_left_weight, int *g_right_weight, int *g_down_weight, int *g_up_weight,
int *g_graph_height, bool *g_pixel_mask, int vertex_num, int width, int height,
int vertex_num1, int width1, int height1, bool *g_over, int *g_counter);

__device__
void add_edge(int from, int to, int cap, int rev_cap, int type, int *d_left_weight,
int *d_right_weight, int *d_down_weight, int *d_up_weight);

__device__
void add_tweights(int i, int cap_source, int  cap_sink, int *d_push_reser, int *d_sink_weight);

__device__
void add_term1(int i, int A, int B, int *d_push_reser, int *d_sink_weight);

__device__
void add_t_links_Cue(int alpha_label, int thid, int *d_left_weight, int *d_right_weight,
int *d_down_weight, int *d_up_weight, int *d_push_reser, int *d_sink_weight,
int *dPixelLabel, int *dDataTerm, int width, int height, int num_labels);

__device__
void add_t_links(int alpha_label, int thid, int *d_left_weight, int *d_right_weight,
int *d_down_weight, int *d_up_weight, int *d_push_reser, int *d_sink_weight,
int *dPixelLabel, int *dDataTerm, int width, int height, int num_labels);

__device__
void add_term2(int x, int y, int A, int B, int C, int D, int type, int *d_left_weight,
int *d_right_weight, int *d_down_weight, int *d_up_weight, int *d_push_reser, int *d_sink_weight);

__device__
void set_up_expansion_energy_G_ARRAY(int alpha_label, int thid, int *d_left_weight, int *d_right_weight,
int *d_down_weight, int *d_up_weight, int *d_push_reser,
int *d_sink_weight, int *dPixelLabel, int *dDataTerm, int *dSmoothTerm,
int width, int height, int num_labels);

__device__
void set_up_expansion_energy_G_ARRAY_Cue(int alpha_label, int thid, int *d_left_weight, int *d_right_weight,
int *d_down_weight, int *d_up_weight, int *d_push_reser,
int *d_sink_weight, int *dPixelLabel, int *dDataTerm, int *dSmoothTerm,
int *dHcue, int *dVcue, int width, int height, int num_labels);



__global__
void CudaWeightCue(int alpha_label, int *d_left_weight, int *d_right_weight, int *d_down_weight,
int *d_up_weight, int *d_push_reser, int *d_sink_weight, int *dPixelLabel,
int *dDataTerm, int *dSmoothTerm, int *dHcue, int *dVcue, int width, int height, int num_labels);


__global__
void CudaWeight(int alpha_label, int *d_left_weight, int *d_right_weight, int *d_down_weight,
int *d_up_weight, int *d_push_reser, int *d_sink_weight, int *dPixelLabel,
int *dDataTerm, int *dSmoothTerm, int width, int height, int num_labels);

__global__
void adjustedgeweight(int *d_left_weight, int *d_right_weight, int *d_down_weight, int *d_up_weight,
int *d_push_reser, int *d_sink_weight, int *temp_left_weight, int *temp_right_weight,
int *temp_down_weight, int *temp_up_weight, int *temp_push_reser, int *temp_sink_weight,
int width, int height, int graph_size, int width1, int height1, int graph_size1);

__global__
void copyedgeweight(int *d_left_weight, int *d_right_weight, int *d_down_weight, int *d_up_weight,
int *d_push_reser, int *d_sink_weight, int *temp_left_weight, int *temp_right_weight,
int *temp_down_weight, int *temp_up_weight, int *temp_push_reser, int *temp_sink_weight,
int *d_pull_left, int *d_pull_right, int *d_pull_down, int *d_pull_up, int *d_relabel_mask,
int *d_graph_heightr, int *d_graph_heightw, int width, int height, int graph_size, int width1, int height1, int graph_size1);

#endif
