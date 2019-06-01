#include "GraphCuts.h"

int GraphCuts::graphCutsInit(int widthGrid, int heightGrid, int labels)
{
	deviceCount = checkDevice();

	printf("No. of devices %d\n", deviceCount);
	if (deviceCount < 1)
		return -1;

	int cuda_device = 0;

	cudaSetDevice(cuda_device);

	cudaDeviceProp device_properties;

	(cudaGetDeviceProperties(&device_properties, cuda_device));

	if ((3 <= device_properties.major) && (device_properties.minor < 1))
		deviceCheck = 2;
	else
	if ((3 <= device_properties.major) && (device_properties.minor >= 1))
		deviceCheck = 1;
	else
		deviceCheck = 0;

	width = widthGrid;
	height = heightGrid;
	num_Labels = labels;

	blocks_x = 1;
	blocks_y = 1;
	num_of_blocks = 1;

	num_of_threads_per_block = 256;
	threads_x = 32;
	threads_y = 8;

	width1 = threads_x * ((int)ceil((float)width / (float)threads_x));
	height1 = threads_y * ((int)ceil((float)height / (float)threads_y));

	graph_size = width * height;
	graph_size1 = width1 * height1;
	size_int = sizeof(int)* graph_size1;

	blocks_x = (int)((ceil)((float)width1 / (float)threads_x));
	blocks_y = (int)((ceil)((float)height1 / (float)threads_y));

	num_of_blocks = (int)((ceil)((float)graph_size1 / (float)num_of_threads_per_block));

	h_mem_init();
	d_mem_init();
	cueValues = 0;

	return deviceCheck;
}


int GraphCuts::checkDevice()
{
	int deviceCount;

	cudaGetDeviceCount(&deviceCount);

	if (deviceCount == 0)
	{
		return -1;
	}


	return deviceCount;
}

void GraphCuts::h_mem_init()
{
	h_reset_mem = (int*)malloc(sizeof(int)* graph_size1);
	h_graph_height = (int*)malloc(size_int);
	pixelLabel = (int*)malloc(size_int);
	h_pixel_mask = (bool*)malloc(sizeof(bool)* graph_size1);

	for (int i = 0; i < graph_size1; i++)
	{
		pixelLabel[i] = 0;
		h_graph_height[i] = 0;
	}

	for (int i = 0; i < graph_size1; i++)
	{
		h_reset_mem[i] = 0;
	}
}


void GraphCuts::d_mem_init()
{
	cudaMalloc((void**)&d_left_weight, sizeof(int)* graph_size1);
	cudaMalloc((void**)&d_right_weight, sizeof(int)* graph_size1);
	cudaMalloc((void**)&d_down_weight, sizeof(int)* graph_size1);
	cudaMalloc((void**)&d_up_weight, sizeof(int)* graph_size1);
	cudaMalloc((void**)&d_push_reser, sizeof(int)* graph_size1);
	cudaMalloc((void**)&d_sink_weight, sizeof(int)* graph_size1);

	cudaMalloc((void**)&s_left_weight, sizeof(int)* graph_size1);
	cudaMalloc((void**)&s_right_weight, sizeof(int)* graph_size1);
	cudaMalloc((void**)&s_down_weight, sizeof(int)* graph_size1);
	cudaMalloc((void**)&s_up_weight, sizeof(int)* graph_size1);
	cudaMalloc((void**)&s_push_reser, sizeof(int)* graph_size1);
	cudaMalloc((void**)&s_sink_weight, sizeof(int)* graph_size1);

	(cudaMalloc((void**)&d_stochastic, sizeof(int)* num_of_blocks));
	(cudaMalloc((void**)&d_stochastic_pixel, sizeof(int)* graph_size1));
	(cudaMalloc((void**)&d_terminate, sizeof(int)* num_of_blocks));


	(cudaMalloc((void**)&d_pull_left, sizeof(int)* graph_size1));
	(cudaMalloc((void**)&d_pull_right, sizeof(int)* graph_size1));
	(cudaMalloc((void**)&d_pull_down, sizeof(int)* graph_size1));
	(cudaMalloc((void**)&d_pull_up, sizeof(int)* graph_size1));

	(cudaMalloc((void**)&d_graph_heightr, sizeof(int)* graph_size1));
	(cudaMalloc((void**)&d_graph_heightw, sizeof(int)* graph_size1));
	(cudaMalloc((void**)&d_relabel_mask, sizeof(int)* graph_size1));

	(cudaMalloc((void**)&d_pixel_mask, sizeof(bool)*graph_size1));
	(cudaMalloc((void**)&d_over, sizeof(bool)* 1));
	(cudaMalloc((void**)&d_counter, sizeof(int)));

	(cudaMalloc((void **)&dPixelLabel, sizeof(int)* width1 * height1));
	(cudaMemcpy(d_left_weight, h_reset_mem, sizeof(int)* graph_size1, cudaMemcpyHostToDevice));
	(cudaMemcpy(d_right_weight, h_reset_mem, sizeof(int)* graph_size1, cudaMemcpyHostToDevice));
	(cudaMemcpy(d_down_weight, h_reset_mem, sizeof(int)* graph_size1, cudaMemcpyHostToDevice));
	(cudaMemcpy(d_up_weight, h_reset_mem, sizeof(int)* graph_size1, cudaMemcpyHostToDevice));
	(cudaMemcpy(d_push_reser, h_reset_mem, sizeof(int)* graph_size1, cudaMemcpyHostToDevice));
	(cudaMemcpy(d_sink_weight, h_reset_mem, sizeof(int)* graph_size1, cudaMemcpyHostToDevice));

	h_relabel_mask = (int*)malloc(sizeof(int)*width1*height1);

	h_stochastic = (int *)malloc(sizeof(int)* num_of_blocks);
	h_stochastic_pixel = (int *)malloc(sizeof(int)* graph_size1);



	for (int i = 0; i < graph_size1; i++)
		h_relabel_mask[i] = 1;


	(cudaMemcpy(d_relabel_mask, h_relabel_mask, sizeof(int)* graph_size1, cudaMemcpyHostToDevice));

	int *dpixlab = (int*)malloc(sizeof(int)*width1*height1);

	for (int i = 0; i < width1 * height1; i++)
	{
		dpixlab[i] = 0;
		h_stochastic_pixel[i] = 1;
	}

	for (int i = 0; i < num_of_blocks; i++)
	{
		h_stochastic[i] = 1;
	}

	(cudaMemcpy(d_stochastic, h_stochastic, sizeof(int)* num_of_blocks, cudaMemcpyHostToDevice));
	(cudaMemcpy(d_terminate, h_stochastic, sizeof(int)* num_of_blocks, cudaMemcpyHostToDevice));
	(cudaMemcpy(d_stochastic_pixel, h_stochastic_pixel, sizeof(int)* graph_size1, cudaMemcpyHostToDevice));


	(cudaMemcpy(dPixelLabel, dpixlab, sizeof(int)* width1 * height1, cudaMemcpyHostToDevice));

	free(dpixlab);
}

int GraphCuts::graphCutsSetupDataTerm()
{
	if (deviceCheck < 1)
		return -1;

	(cudaMalloc((void **)&dDataTerm, sizeof(int)* width * height * num_Labels));

	(cudaMemcpy(dDataTerm, dataTerm, sizeof(int)* width * height * num_Labels, cudaMemcpyHostToDevice));

	return 0;
}


int GraphCuts::graphCutsSetupSmoothTerm()
{
	if (deviceCheck < 1)
		return -1;

	(cudaMalloc((void **)&dSmoothTerm, sizeof(int)* num_Labels * num_Labels));

	(cudaMemcpy(dSmoothTerm, smoothTerm, sizeof(int)* num_Labels * num_Labels, cudaMemcpyHostToDevice));

	return 0;
}

int GraphCuts::graphCutsSetupHCue()
{

	if (deviceCheck < 1)
		return -1;

	(cudaMalloc((void **)&dHcue, sizeof(int)* width * height));

	(cudaMemcpy(dHcue, hCue, sizeof(int)* width * height, cudaMemcpyHostToDevice));

	cueValues = 1;

	return 0;
}

int GraphCuts::graphCutsSetupVCue()
{
	if (deviceCheck < 1)
		return -1;

	(cudaMalloc((void **)&dVcue, sizeof(int)* width * height));

	(cudaMemcpy(dVcue, vCue, sizeof(int)* width * height, cudaMemcpyHostToDevice));

	return 0;
}

int GraphCuts::graphCutsSetupGraph()
{

	if (deviceCheck < 1)
		return -1;

	int alpha_label = 1;

	for (int i = 0; i < graph_size1; i++)
	{
		h_reset_mem[i] = 0;
		h_graph_height[i] = 0;
	}

	int blockEdge = (int)((ceil)((float)(width * height) / (float)256));
	dim3 block_weight(256, 1, 1);
	dim3 grid_weight(blockEdge, 1, 1);

	if (cueValues == 1)
	{
		CudaWeightCue <<< grid_weight, block_weight >>>(alpha_label, d_left_weight, d_right_weight, d_down_weight,
			d_up_weight, d_push_reser, d_sink_weight, dPixelLabel, dDataTerm,
			dSmoothTerm, dHcue, dVcue, width, height, 2);
	}
	else
	{
		CudaWeight <<< grid_weight, block_weight >>>(alpha_label, d_left_weight, d_right_weight, d_down_weight,
			d_up_weight, d_push_reser, d_sink_weight, dPixelLabel, dDataTerm,
			dSmoothTerm, width, height, 2);

	}

	int *temp_left_weight, *temp_right_weight, *temp_down_weight, *temp_up_weight, *temp_source_weight, *temp_terminal_weight;

	(cudaMalloc((void **)&temp_left_weight, sizeof(int)* graph_size1));
	(cudaMalloc((void **)&temp_right_weight, sizeof(int)* graph_size1));
	(cudaMalloc((void **)&temp_down_weight, sizeof(int)* graph_size1));
	(cudaMalloc((void **)&temp_up_weight, sizeof(int)* graph_size1));
	(cudaMalloc((void **)&temp_source_weight, sizeof(int)* graph_size1));
	(cudaMalloc((void **)&temp_terminal_weight, sizeof(int)* graph_size1));

	int blockEdge1 = (int)((ceil)((float)(width1 * height1) / (float)256));
	dim3 block_weight1(256, 1, 1);
	dim3 grid_weight1(blockEdge1, 1, 1);

	adjustedgeweight <<<grid_weight1, block_weight1 >>>(d_left_weight, d_right_weight, d_down_weight, d_up_weight, d_push_reser,
		d_sink_weight, temp_left_weight, temp_right_weight, temp_down_weight, temp_up_weight,
		temp_source_weight, temp_terminal_weight, width, height, graph_size, width1,
		height1, graph_size1);

	copyedgeweight <<<grid_weight1, block_weight1 >>>(d_left_weight, d_right_weight, d_down_weight, d_up_weight, d_push_reser, d_sink_weight,
		temp_left_weight, temp_right_weight, temp_down_weight, temp_up_weight, temp_source_weight,
		temp_terminal_weight, d_pull_left, d_pull_right, d_pull_down, d_pull_up, d_relabel_mask,
		d_graph_heightr, d_graph_heightw, width, height, graph_size, width1, height1, graph_size1);

	(cudaFree(temp_left_weight));
	(cudaFree(temp_right_weight));
	(cudaFree(temp_up_weight));
	(cudaFree(temp_down_weight));
	(cudaFree(temp_source_weight));
	(cudaFree(temp_terminal_weight));
	return 0;
}

int GraphCuts::graphCutsAtomicOptimize()
{
	if (deviceCheck < 1)
	{
		return -1;
	}

	graphCutsAtomic();

	bfsLabeling();

	return 0;

}

int GraphCuts::graphCutsStochasticOptimize()
{
	if (deviceCheck < 1)
	{
		return -1;
	}

	graphCutsStochastic();

	bfsLabeling();

	return 0;

}

void GraphCuts::graphCutsAtomic()
{

	dim3 block_push(threads_x, threads_y, 1);
	dim3 grid_push(blocks_x, blocks_y, 1);

	dim3 d_block(num_of_threads_per_block, 1, 1);
	dim3 d_grid(num_of_blocks, 1, 1);

	bool finish = true;

	counter = num_of_blocks;

	int numThreadsEnd = 256, numBlocksEnd = 1;
	if (numThreadsEnd > counter)
	{
		numBlocksEnd = 1;
		numThreadsEnd = counter;
	}
	else
	{
		numBlocksEnd = (int)ceil(counter / (double)numThreadsEnd);
	}

	dim3 End_block(numThreadsEnd, 1, 1);
	dim3 End_grid(numBlocksEnd, 1, 1);

	int *d_counter;

	bool *d_finish;
	for (int i = 0; i < num_of_blocks; i++)
	{
		h_stochastic[i] = 0;
	}

	(cudaMalloc((void**)&d_counter, sizeof(int)));
	(cudaMalloc((void**)&d_finish, sizeof(bool)));

	(cudaMemcpy(d_counter, &counter, sizeof(int), cudaMemcpyHostToDevice));

	counter = 0;
	int *d_relabel;

	(cudaMalloc((void**)&d_relabel, sizeof(int)));

	int h_relabel = 0;

	int block_num = width1 / 32;

	int *d_block_num;

	(cudaMalloc((void**)&d_block_num, sizeof(int)));
	(cudaMemcpy(d_block_num, &block_num, sizeof(int), cudaMemcpyHostToDevice));

	int h_count_blocks = num_of_blocks;
	int *d_count_blocks;

	(cudaMalloc((void**)&d_count_blocks, sizeof(int)));
	(cudaMemcpy(d_count_blocks, &h_count_blocks, sizeof(int), cudaMemcpyHostToDevice));

	h_count_blocks = 0;



	(cudaMemcpy(d_relabel, &h_relabel, sizeof(int), cudaMemcpyHostToDevice));

	counter = 1;
	kernel_push1_start_atomic <<<grid_push, block_push >>>(d_left_weight, d_right_weight, d_down_weight, d_up_weight,
		d_sink_weight, d_push_reser,
		d_relabel_mask, d_graph_heightr, d_graph_heightw, graph_size, width, height,
		graph_size1, width1, height1, d_relabel, d_stochastic, d_counter, d_finish);

	int h_terminate_condition = 0;
	(cudaDeviceSynchronize());
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	do
	{

		if (counter % 10 == 0)
		{
			finish = true;
			(cudaMemcpy(d_finish, &finish, sizeof(bool), cudaMemcpyHostToDevice));
			kernel_push_stochastic1 <<<grid_push, block_push >>>(d_push_reser, s_push_reser, d_count_blocks, d_finish, d_block_num, width1);
			(cudaMemcpy(&finish, d_finish, sizeof(bool), cudaMemcpyDeviceToHost));
			if (finish == false)
				h_terminate_condition++;
		}
		if (counter % 11 == 0)
		{
			(cudaMemset(d_terminate, 0, sizeof(int)*num_of_blocks));
			h_count_blocks = 0;
			(cudaMemcpy(d_count_blocks, &h_count_blocks, sizeof(int), cudaMemcpyHostToDevice));
			kernel_push_atomic2 <<<grid_push, block_push >>>(d_terminate, d_push_reser, s_push_reser, d_block_num, width1);

			kernel_End <<<End_grid, End_block >>>(d_terminate, d_count_blocks, d_counter);

		}

		if (counter % 2 == 0)
		{

			kernel_push1_atomic <<<grid_push, block_push >>>(d_left_weight, d_right_weight, d_down_weight, d_up_weight,
				d_sink_weight, d_push_reser, d_pull_left, d_pull_right, d_pull_down, d_pull_up,
				d_relabel_mask, d_graph_heightr, d_graph_heightw, graph_size, width, height,
				graph_size1, width1, height1);

			kernel_relabel_atomic <<<grid_push, block_push >>>(d_left_weight, d_right_weight, d_down_weight, d_up_weight,
				d_sink_weight, d_push_reser, d_pull_left, d_pull_right, d_pull_down, d_pull_up,
				d_relabel_mask, d_graph_heightr, d_graph_heightw, graph_size, width, height,
				graph_size1, width1, height1);
		}
		else
		{
			kernel_push1_atomic <<<grid_push, block_push >>>(d_left_weight, d_right_weight, d_down_weight, d_up_weight,
				d_sink_weight, d_push_reser, d_pull_left, d_pull_right, d_pull_down, d_pull_up,
				d_relabel_mask, d_graph_heightw, d_graph_heightr, graph_size, width, height,
				graph_size1, width1, height1);
			kernel_relabel_atomic <<<grid_push, block_push >>>(d_left_weight, d_right_weight, d_down_weight, d_up_weight,
				d_sink_weight, d_push_reser, d_pull_left, d_pull_right, d_pull_down, d_pull_up,
				d_relabel_mask, d_graph_heightw, d_graph_heightr, graph_size, width, height,
				graph_size1, width1, height1);

		}
		counter++;
	} while (h_terminate_condition != 2);

	(cudaEventRecord(stop, 0));
	(cudaEventSynchronize(stop));

}


void GraphCuts::graphCutsStochastic()
{

	dim3 block_push(threads_x, threads_y, 1);
	dim3 grid_push(blocks_x, blocks_y, 1);

	dim3 d_block(num_of_threads_per_block, 1, 1);
	dim3 d_grid(num_of_blocks, 1, 1);

	bool finish = true;

	counter = num_of_blocks;

	int numThreadsEnd = 256, numBlocksEnd = 1;
	if (numThreadsEnd > counter)
	{
		numBlocksEnd = 1;
		numThreadsEnd = counter;
	}
	else
	{
		numBlocksEnd = (int)ceil(counter / (double)numThreadsEnd);
	}

	dim3 End_block(numThreadsEnd, 1, 1);
	dim3 End_grid(numBlocksEnd, 1, 1);




	bool *d_finish;
	for (int i = 0; i < num_of_blocks; i++)
	{
		h_stochastic[i] = 0;
	}

	(cudaMalloc((void**)&d_counter, sizeof(int)));
	(cudaMalloc((void**)&d_finish, sizeof(bool)));

	(cudaMemcpy(d_counter, &counter, sizeof(int), cudaMemcpyHostToDevice));

	counter = 0;
	int *d_relabel;

	(cudaMalloc((void**)&d_relabel, sizeof(int)));

	int h_relabel = 0;


	int block_num = width1 / 32;

	int *d_block_num;

	(cudaMalloc((void**)&d_block_num, sizeof(int)));
	(cudaMemcpy(d_block_num, &block_num, sizeof(int), cudaMemcpyHostToDevice));


	int h_count_blocks = num_of_blocks;
	int *d_count_blocks;

	(cudaMalloc((void**)&d_count_blocks, sizeof(int)));
	(cudaMemcpy(d_count_blocks, &h_count_blocks, sizeof(int), cudaMemcpyHostToDevice));

	h_count_blocks = 0;

	(cudaMemcpy(d_relabel, &h_relabel, sizeof(int), cudaMemcpyHostToDevice));

	counter = 1;
	kernel_push1_start_stochastic << <grid_push, block_push >> >(d_left_weight, d_right_weight, d_down_weight, d_up_weight,
		d_sink_weight, d_push_reser,
		d_relabel_mask, d_graph_heightr, d_graph_heightw, graph_size, width, height,
		graph_size1, width1, height1, d_relabel, d_stochastic, d_counter, d_finish);
	int h_terminate_condition = 0;
	(cudaDeviceSynchronize());
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	do
	{
		if (counter % 10 == 0)
		{
			finish = true;
			(cudaMemcpy(d_finish, &finish, sizeof(bool), cudaMemcpyHostToDevice));
			kernel_push_stochastic1 << <grid_push, block_push >> >(d_push_reser, s_push_reser, d_count_blocks, d_finish, d_block_num, width1);
			(cudaMemcpy(&finish, d_finish, sizeof(bool), cudaMemcpyDeviceToHost));
		}
		if (counter % 11 == 0)
		{
			(cudaMemset(d_stochastic, 0, sizeof(int)*num_of_blocks));
			(cudaMemset(d_terminate, 0, sizeof(int)*num_of_blocks));
			h_count_blocks = 0;
			(cudaMemcpy(d_count_blocks, &h_count_blocks, sizeof(int), cudaMemcpyHostToDevice));
			kernel_push_stochastic2 << <grid_push, block_push >> >(d_terminate, d_relabel_mask, d_push_reser, s_push_reser, d_stochastic, d_block_num, width1);

			kernel_End << <End_grid, End_block >> >(d_terminate, d_count_blocks, d_counter);

			if (finish == false && counter % 121 != 0 && counter > 0)
				h_terminate_condition++;

		}
		if (counter % 2 == 0)
		{

			kernel_push1_stochastic << <grid_push, block_push >> >(d_left_weight, d_right_weight, d_down_weight, d_up_weight,
				d_sink_weight, d_push_reser,
				d_relabel_mask, d_graph_heightr, d_graph_heightw, graph_size, width, height,
				graph_size1, width1, height1, d_stochastic, d_block_num);

			kernel_relabel_stochastic << <grid_push, block_push >> >(d_left_weight, d_right_weight, d_down_weight, d_up_weight,
				d_sink_weight, d_push_reser,
				d_relabel_mask, d_graph_heightr, d_graph_heightw, graph_size, width, height,
				graph_size1, width1, height1, d_stochastic, d_block_num);

		}
		else
		{
			kernel_push1_stochastic << <grid_push, block_push >> >(d_left_weight, d_right_weight, d_down_weight, d_up_weight,
				d_sink_weight, d_push_reser,
				d_relabel_mask, d_graph_heightw, d_graph_heightr, graph_size, width, height,
				graph_size1, width1, height1, d_stochastic, d_block_num);

			kernel_relabel_stochastic << <grid_push, block_push >> >(d_left_weight, d_right_weight, d_down_weight, d_up_weight,
				d_sink_weight, d_push_reser,
				d_relabel_mask, d_graph_heightw, d_graph_heightr, graph_size, width, height,
				graph_size1, width1, height1, d_stochastic, d_block_num);

		}
		counter++;
	} while (h_terminate_condition == 0 && counter < 500);


	(cudaEventRecord(stop, 0));
	(cudaEventSynchronize(stop));
	float time;
	(cudaEventElapsedTime(&time, start, stop));
	printf("TT Cuts :: %f ms\n", time);

}

void GraphCuts::bfsLabeling()
{

	dim3 block_push(threads_x, threads_y, 1);
	dim3 grid_push(blocks_x, blocks_y, 1);

	dim3 d_block(num_of_threads_per_block, 1, 1);
	dim3 d_grid(num_of_blocks, 1, 1);

	(cudaMemcpy(d_graph_heightr, h_graph_height, size_int, cudaMemcpyHostToDevice));

	for (int i = 0; i < graph_size; i++)
		h_pixel_mask[i] = true;

	(cudaMemcpy(d_pixel_mask, h_pixel_mask, sizeof(bool)* graph_size1, cudaMemcpyHostToDevice));

	kernel_bfs_t << <d_grid, d_block, 0 >> >(d_push_reser, d_sink_weight, d_graph_heightr, d_pixel_mask, graph_size, width, height, graph_size1, width1, height1);
	counter = 1;

	(cudaMemcpy(d_counter, &counter, sizeof(int), cudaMemcpyHostToDevice));

	do
	{
		h_over = false;

		(cudaMemcpy(d_over, &h_over, sizeof(bool), cudaMemcpyHostToDevice));

		kernel_bfs << < d_grid, d_block, 0 >> >(d_left_weight, d_right_weight, d_down_weight, d_up_weight, d_graph_heightr, d_pixel_mask,
			graph_size, width, height, graph_size1, width1, height1, d_over, d_counter);

		(cudaMemcpy(&h_over, d_over, sizeof(bool), cudaMemcpyDeviceToHost));

		counter++;

		(cudaMemcpy(d_counter, &counter, sizeof(int), cudaMemcpyHostToDevice));
	} while (h_over);

	(cudaMemcpy(h_graph_height, d_graph_heightr, size_int, cudaMemcpyDeviceToHost));
}


int GraphCuts::graphCutsGetResult()
{
	if (deviceCheck < 1)
		return -1;

	int alpha = 1;

	for (int i = 0; i < graph_size1; i++)
	{
		int row_here = i / width1, col_here = i % width1;
		if (h_graph_height[i]>0 && row_here < height && row_here > 0 && col_here < width && col_here > 0) {
			pixelLabel[i] = alpha;
		}
	}

	return 0;

}

void GraphCuts::graphCutsFreeMem()
{
	free(h_reset_mem);
	free(h_graph_height);
	free(pixelLabel);
	free(h_pixel_mask);

	free(h_relabel_mask);
	free(h_stochastic);
	free(h_stochastic_pixel);

	free(hCue);
	free(vCue);
	free(dataTerm);
	free(smoothTerm);

	(cudaFree(d_left_weight));
	(cudaFree(d_right_weight));
	(cudaFree(d_down_weight));
	(cudaFree(d_up_weight));
	(cudaFree(d_sink_weight));
	(cudaFree(d_push_reser));

	(cudaFree(d_pull_left));
	(cudaFree(d_pull_right));
	(cudaFree(d_pull_down));
	(cudaFree(d_pull_up));

	(cudaFree(d_graph_heightr));
	(cudaFree(d_graph_heightw));

	(cudaFree(s_left_weight));
	(cudaFree(s_right_weight));
	(cudaFree(s_down_weight));
	(cudaFree(s_up_weight));
	(cudaFree(s_push_reser));
	(cudaFree(s_sink_weight));


	(cudaFree(d_stochastic));
	(cudaFree(d_stochastic_pixel));
	(cudaFree(d_terminate));

	(cudaFree(d_relabel_mask));

	(cudaFree(d_pixel_mask));
	(cudaFree(d_over));
	(cudaFree(d_counter));

	(cudaFree(dPixelLabel));
}
