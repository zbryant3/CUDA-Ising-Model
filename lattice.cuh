//******************************************************************************
//  Author: Zachariah Bryant
//  Function: Creates a lattice object to perform monte carlo Ising Model
//      operations using a NVIDIA GPU with CUDA.
//
//      The lattice size MUST be a multiple of 8 for now as the
//      sublattices are hard coded with this in mind (look in .cu file).
//
//******************************************************************************

#ifndef LATTICE_H
#define LATTICE_H

#include <iostream>
#include <vector>
#include <random>


using namespace std;

struct variables{
  //Lattice variables
  int size;
  double j;
  double beta;
  double h;

  //Constructor and destructor defined in .cu
  variables(int, double, double, double);
};





//************************
//    Device Functions   *
//************************


//Looks down in a given dimension, if the dimension is too small it returns
// the size of the lattice -1, giving the lattice a periodic nature
__device__ int LookDown(variables, int);



//Looks up in a given dimnension, if the dimension is too large it returns a
//0 - giving our lattice a periodic nature
__device__ int LookUp(variables, int);



//Populates the sublattice from the major lattice
__device__ void PopulateSubLattice(variables, int, int*);



//Gets the difference in energy if spin is changed
__device__ double EnergyDiff(variables, int*);

//Returns the Boltzmann distribution of a given energy difference
__device__ float BoltzmannDist(variables, double);

//Equilibriates a given sublattice
__device__ void ThreadEquilibriate(variables, int*);



__global__ void GPU_Equilibriate(variables, int[]);



//******************************
//    Class for Ising model    *
//******************************
class ising_model
{
private:
  variables *host_mem;
  variables *gpu_mem;
  int *host_major_lattice;
  int *gpu_major_lattice;


  //Equilibrates sublattices on the gpu
  //__global__ void GPU_Equilibriate(std::vector<std::vector<int>>);







public:

  //Constructor - Creates the lattice in the host memory and the device memory
  //Input Parameters - Lattice Dimension,
  __host__ ising_model(int, double, double, double);


  //Destructor - Deletes dynamically allocated memory
  __host__ ~ising_model();


  //Gets the average spin of the lattice using the gpu
  __host__ double AverageSpin();

  __host__ void SetBeta(double newbeta){
    host_mem->beta = newbeta;
    cudaMemcpy(gpu_mem, host_mem, sizeof(variables), cudaMemcpyHostToDevice);
  }


  //Equilibrates the lattice
  __host__ void Equilibrate();



};

#endif
