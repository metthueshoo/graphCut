# graphCuts

GraphCut Algorithm based on the paper written by Vibhav Vineet and P.J. Narayanan published in "CUDA Cuts: Fast Graph Cuts on the GPU"

Loosley based on the implementation as given in https://github.com/15pengyi/JF-Cut/tree/master/benchmark/CudaCuts

## input

The first line needs to contain the Width, Length and Height of the image

The next Height * Length lines contains Width integers between label vertex.

The next Height lines contains Width-1 integers describing the horizontal edges.

The next Height-1 lines contains Width integers describing the vertical edges.

See in/ folder for example inputs
 
## output

The output is stored as a grayscale out.pgm in out/
