#include "GraphCuts.h"

using namespace std;

GraphCuts cuts;

void loadFile(char *filename);
void writePGM(char *filename);

int main()
{
	loadFile("in/sponge.txt");

	int initCheck = cuts.graphCutsInit(cuts.width, cuts.height, cuts.num_Labels);

	printf("Compute Capability %d\n", initCheck);

	if (initCheck > 0)
	{
		printf("The grid is initialized successfully\n");
	}
	else
	if (initCheck == -1)
	{
		printf("Error: Please check the device present on the system\n");
	}

	int dataCheck = cuts.graphCutsSetupDataTerm();

	if (dataCheck == 0)
	{
		printf("The dataterm is set properly\n");

	}
	else
	if (dataCheck == -1)
	{
		printf("Error: Please check the device present on the system\n");
	}


	int smoothCheck = cuts.graphCutsSetupSmoothTerm();


	if (smoothCheck == 0)
	{
		printf("The smoothnessterm is set properly\n");
	}
	else
	if (smoothCheck == -1)
	{
		printf("Error: Please check the device present on the system\n");
	}


	int hcueCheck = cuts.graphCutsSetupHCue();

	if (hcueCheck == 0)
	{
		printf("The HCue is set properly\n");
	}
	else
	if (hcueCheck == -1)
	{
		printf("Error: Please check the device present on the system\n");
	}

	int vcueCheck = cuts.graphCutsSetupVCue();


	if (vcueCheck == 0)
	{
		printf("The VCue is set properly\n");
	}
	else
	if (vcueCheck == -1)
	{
		printf("Error: Please check the device present on the system\n");
	}


	int graphCheck = cuts.graphCutsSetupGraph();

	if (graphCheck == 0)
	{
		printf("The graph is constructed successfully\n");
	}
	else
	if (graphCheck == -1)
	{
		printf("Error: Please check the device present on the system\n");
	}

	int optimizeCheck = -1;
	if (initCheck == 1)
	{
		optimizeCheck = cuts.graphCutsStochasticOptimize();
	}

	if (optimizeCheck == 0)
	{
		printf("The algorithm successfully converged\n");
	}
	else
	if (optimizeCheck == -1)
	{
		printf("Error: Please check the device present on the system\n");
	}

	int resultCheck = cuts.graphCutsGetResult();

	if (resultCheck == 0)
	{
		printf("The pixel labels are successfully stored\n");
	}
	else
	if (resultCheck == -1)
	{
		printf("Error: Please check the device present on the system\n");
	}

	writePGM("out.pgm");

	cuts.graphCutsFreeMem();

	return 0;
}

void writePGM(char* filename)
{
	int** out_pixel_values = (int**)malloc(sizeof(int*)*cuts.height);

	for (int i = 0; i < cuts.height; i++) {
		out_pixel_values[i] = (int*)malloc(sizeof(int)* cuts.width);
		for (int j = 0; j < cuts.width; j++) {
			out_pixel_values[i][j] = 0;
		}
	}

	for (int i = 0; i < cuts.graph_size1; i++) {
		int row = i / cuts.width1, col = i % cuts.width1;
		if (row >= 0 && col >= 0 && row <= cuts.height - 1 && col <= cuts.width - 1)
			out_pixel_values[row][col] = cuts.pixelLabel[i] * 255;
	}

	FILE* fp = fopen(filename, "w");

	fprintf(fp, "%c", 'P');
	fprintf(fp, "%c", '2');
	fprintf(fp, "%c", '\n');
	fprintf(fp, "%d %c %d %c ", cuts.width, ' ', cuts.height, '\n');
	fprintf(fp, "%d %c", 255, '\n');

	for (int i = 0; i<cuts.height; i++)
	{
		for (int j = 0; j<cuts.width; j++)
		{
			fprintf(fp, "%d\n", out_pixel_values[i][j]);
		}
	}
	fclose(fp);
	for (int i = 0; i < cuts.height; i++)
		free(out_pixel_values[i]);
	free(out_pixel_values);
}


void loadFile(char *filename)
{
	int &width = cuts.width;
	int &height = cuts.height;
	int &nLabels = cuts.num_Labels;

	int *&dataCostArray = cuts.dataTerm;
	int *&smoothCostArray = cuts.smoothTerm;
	int *&hCue = cuts.hCue;
	int *&vCue = cuts.vCue;

	FILE *fp = fopen(filename, "r");

	fscanf(fp, "%d %d %d", &width, &height, &nLabels);

	int i, n, x, y;
	int gt;
	for (i = 0; i < width * height; i++)
		fscanf(fp, "%d", &gt);

	dataCostArray = (int*)malloc(sizeof(int)* width * height * nLabels);

	for (int c = 0; c < nLabels; c++) {
		n = c;
		for (i = 0; i < width * height; i++) {
			fscanf(fp, "%d", &dataCostArray[n]);
			n += nLabels;
		}
	}

	hCue = (int*)malloc(sizeof(int)* width * height);
	vCue = (int*)malloc(sizeof(int)* width * height);

	n = 0;
	for (y = 0; y < height; y++) {
		for (x = 0; x < width - 1; x++) {
			fscanf(fp, "%d", &hCue[n++]);
		}
		hCue[n++] = 0;
	}

	n = 0;
	for (y = 0; y < height - 1; y++) {
		for (x = 0; x < width; x++) {
			fscanf(fp, "%d", &vCue[n++]);
		}
	}

	for (x = 0; x < width; x++) {
		vCue[n++] = 0;
	}

	fclose(fp);
	smoothCostArray = (int*)malloc(sizeof(int)*nLabels * nLabels);
	smoothCostArray[0] = 0;
	smoothCostArray[1] = 1;
	smoothCostArray[2] = 1;
	smoothCostArray[3] = 0;
}
