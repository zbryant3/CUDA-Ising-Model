#include "lattice.cuh"

#include <iostream>
#include <vector>


#include <cuda.h>
#include <curand_kernel.h>


#include <stdio.h> //For testing


using namespace std;


/***************************************
*         Structure Functions          *
***************************************/


//-------------------------------------------------
//    Constructor for inner stucture variables    -
//-------------------------------------------------
variables::variables(int inputSize, double inputBeta, double inputJ, double inputH){

  //Sets variables to prescribed values
  size = inputSize;
  beta = inputBeta;
  j = inputJ;
  h = inputH;

};



/************************************
*         Device Functions          *
************************************/


//-------------------------------------------------------------
//    Looks down in a given dimension, if the dimension is    -
//    too small it returns the size of the lattice -1,        -
//    giving the lattice a periodic nature                    -
//-------------------------------------------------------------
__device__ int LookDown(variables *gpu_mem, int dimension){

  if( (dimension - 1) < 0)
  return (gpu_mem->size - 1);
  else
  return (dimension - 1);
};


//-------------------------------------------------------------
//    Looks up in a given dimnension, if the dimension is     -
//    too large it returns a 0 - giving the lattice           -
//    a periodic nature                                       -
//-------------------------------------------------------------
__device__ int LookUp(variables *gpu_mem, int dimension){

  if( (dimension + 1) >= gpu_mem->size)
  return 0;
  else
  return (dimension + 1);

};


//-------------------------------------------------------------
//    Populates the sublattices by looking at the major       -
//    lattice while perserving the periodicity                -
//-------------------------------------------------------------
__device__ void PopulateSubLattice(variables *gpu_mem, int *major_lattice, int *sub_lattice){

  //Find thread location on major lattice
  int majorX = threadIdx.x + blockIdx.x * blockDim.x;
  int majorY = threadIdx.y + blockIdx.y * blockDim.y;


  //Populate sublatt to major lattice values
  sub_lattice[(threadIdx.x + 1) + (threadIdx.y + 1)*blockDim.y] = major_lattice[majorX + majorY*gpu_mem->size];


  /* For testing the up and down transitions
            uncomment to reuse

  if(majorX == 0 && majorY == 23){
    printf("Original - majorX:  %i \t majorY:  %i \n", majorX, majorY);
    printf("Down -  majorX: %i \t majorY: %i \n", LookDown(gpu_mem, majorX), LookDown(gpu_mem, majorY));
    printf("Up - majorX:  %i \t majorY:  %i \n \n", LookUp(gpu_mem, majorX), LookUp(gpu_mem, majorY));
  }
  */


  //If thread id for y = 0, then look up in major lattice to fill sub lattice
  if(threadIdx.y == 0){
    sub_lattice[(threadIdx.x + 1) + 0*blockDim.y] = major_lattice[ majorX + LookDown(gpu_mem, majorY)*gpu_mem->size];
  }


  //If thread id for y is the max, then look down in the major lattice to fill sub lattice
  if((threadIdx.y + 1) == blockDim.y ){
    sub_lattice[(threadIdx.x + 1) + (threadIdx.y + 2)*blockDim.y] = major_lattice[majorX + LookUp(gpu_mem, majorY)*gpu_mem->size];
  }
  __syncthreads();

  //If thread id for x is 0, then look up in the x direction to fill sub lattice
  if(threadIdx.x == 0){
    sub_lattice[0 + (threadIdx.y + 1)*blockDim.y] = major_lattice[LookDown(gpu_mem, majorX) + majorY*gpu_mem->size ];
  }


  //If thread id for x is the max, then look down in the x direction to fill sub lattice
  if(threadIdx.x == (blockDim.x - 1) ){
    sub_lattice[(blockDim.x + 1) + 0*blockDim.y] = major_lattice[LookUp(gpu_mem, majorX) + majorY*gpu_mem->size];
  }
  __syncthreads();

};


//-------------------------------------------------------------
//    Gets the difference in energy if spin is changed        -
//-------------------------------------------------------------
__device__ double EnergyDiff(variables *gpu_mem, int *lattice){

  int x = threadIdx.x + 1;
  int y = threadIdx.y + 1;
  int spindiff = (-1)*lattice[x + y*blockDim.y] - lattice[x + y*blockDim.y];
  double sumNeighborSpin{0};
  double energydiff{0};

  sumNeighborSpin += lattice[x + (y + 1)*blockDim.y];
  sumNeighborSpin += lattice[x + (y - 1)*blockDim.y];
  sumNeighborSpin += lattice[(x + 1) + y*blockDim.y];
  sumNeighborSpin += lattice[(x - 1) + y*blockDim.y];

  energydiff += (-1)*gpu_mem->j*sumNeighborSpin*spindiff;
  energydiff += (-1)*gpu_mem->h*spindiff;

  return energydiff;
};



//---------------------------------------------------------------
//    Determines the probability of a given energy difference   -
//    of a spin being flipped at a location.                    -
//---------------------------------------------------------------
__device__ float BoltzmannDist(variables *gpu_mem, double energydiff){
  return expf((-1)*gpu_mem->beta*energydiff);
};



//---------------------------------------
//    Equilibrates a given sublattice   -
//---------------------------------------
__device__ void ThreadEquilibriate(variables *gpu_mem, int *lattice){

  //Determines if thread is on an even or odd grid
  // 0 if even, 1 if odd
  int remainder = (threadIdx.x + threadIdx.y) % 2;


  int x = threadIdx.x + 1;
  int y = threadIdx.y + 1;
  int new_spin = (-1)*lattice[x + y*blockDim.y];
  int tid = threadIdx.x + blockIdx.x*blockDim.x + threadIdx.y + blockIdx.y*blockDim.y;
  double energydiff{0};


  curandState_t rng;
  curand_init(clock64(), tid, 0, &rng);

  //Even Grid Equilibration
  if(remainder == 0){
    energydiff = EnergyDiff(gpu_mem, lattice);

    if(energydiff <= 0){
      lattice[x + y*blockDim.y] = new_spin;
    } else if(curand_uniform(&rng) < BoltzmannDist(gpu_mem, energydiff)){
      lattice[x + y*blockDim.y] = new_spin;
    }
  }
  __syncthreads();

  //Odd Grid Equilibration
  if(remainder == 1){
    energydiff = EnergyDiff(gpu_mem, lattice);

    if(energydiff <= 0){
      lattice[x + y*blockDim.y] = new_spin;
    } else if(curand_uniform(&rng) < BoltzmannDist(gpu_mem, energydiff)){
      lattice[x + y*blockDim.y] = new_spin;
    }
  }
  __syncthreads();

};


//-------------------------------------------
//    Equilibrates sublattices on the gpu   -
//-------------------------------------------
__global__ void GPU_Equilibriate(variables *gpu_mem, int *major_lattice){


  //Shared sublattice memory for each block to share among the threads
  __shared__ int sub_lattice[10*10];
  PopulateSubLattice(gpu_mem, major_lattice, sub_lattice);


  //First we will equilibriate even blocks, then the odd blocks.
  int remainder = (blockIdx.y  + blockIdx.x) % 2;


  //Even checkerboard pattern for blocks
  if(remainder == 0){
    ThreadEquilibriate(gpu_mem, sub_lattice);
  }
  __syncthreads();


  //Odd Checkerboard pattern for blocks
  if(remainder == 1){
    ThreadEquilibriate(gpu_mem, sub_lattice);
  }
  __syncthreads();

  //Find thread location on major lattice
  int majorX = threadIdx.x + blockIdx.x * blockDim.x;
  int majorY = threadIdx.y + blockIdx.y * blockDim.y;
  major_lattice[majorX + majorY*gpu_mem->size] = sub_lattice[(threadIdx.x + 1) + (threadIdx.y + 1)*blockDim.y];


};




/*************************************
*         Private Functions          *
*************************************/




/************************************
*         Public Functions          *
************************************/


//-----------------------------------------------------------------
//    Constructor - Creates the lattice in the                    -
//                       host memory and the device memory        -
//    Input Parameters - Lattice dimensions,                      -
//                       starting beta, starting J, starting h.   -
//-----------------------------------------------------------------
__host__ ising_model::ising_model(int inputSize, double inputBeta, double inputJ, double inputH){

  host_mem = new variables(inputSize, inputBeta, inputJ, inputH);
  //Allocates Device memory and copies the host mem to the device
  cudaMalloc(&gpu_mem, sizeof(variables));
  cudaMemcpy(gpu_mem, host_mem, sizeof(variables), cudaMemcpyHostToDevice);

  host_major_lattice = new int[inputSize*inputSize];

  //Set all values of major lattice to one
  for(int x = 0; x < inputSize; x++)
  {
    for(int y = 0; y < inputSize; y++)
    {
      host_major_lattice[x + y*inputSize] = 1;
    }
  }

  //Allocate and copy host_major_lattice to the device
  cudaMalloc(&gpu_major_lattice, sizeof(int)*inputSize*inputSize);
  cudaMemcpy(gpu_major_lattice, host_major_lattice, sizeof(int)*inputSize*inputSize, cudaMemcpyHostToDevice);


};


//---------------------------------------------------------
//    Destructor - Deletes dynamically allocated memory   -
//---------------------------------------------------------
__host__ ising_model::~ising_model(){

  delete host_mem;
  cudaFree(gpu_mem);

  delete[] host_major_lattice;
  cudaFree(gpu_major_lattice);


};


//----------------------------------------------
//  Gets the average spin of the lattice using -
//      host (no need for gpu acceleration)    -
//----------------------------------------------
__host__ double ising_model::AverageSpin(){

  double total{0};

  for(int x = 0; x < host_mem->size; x++)
  for(int y = 0; y < host_mem->size; y++)
  total += host_major_lattice[x + y*host_mem->size];

  return total/(host_mem->size * host_mem->size);

};


//---------------------------------
//    Equilibrates the lattice    -
//---------------------------------
__host__ void ising_model::Equilibrate(){

  //Specify the dimensions of the sublattices
  dim3 threads(8, 8);
  dim3 blocks(host_mem->size/8, host_mem->size/8);


  //Copy Host lattice to Device lattice
  cudaMemcpy(gpu_major_lattice, host_major_lattice, sizeof(int)*host_mem->size*host_mem->size, cudaMemcpyHostToDevice);

  //Use GPU to equilibriate on sublattices
  GPU_Equilibriate<<<blocks, threads>>>(gpu_mem, gpu_major_lattice);
  cudaDeviceSynchronize();


  //Copy Device Memory to Host Memory
  cudaMemcpy(host_major_lattice, gpu_major_lattice, sizeof(int)*host_mem->size*host_mem->size, cudaMemcpyDeviceToHost);

};
