//*****************************************************
// Usage: Performs Ising model simulations utilizing  *
//  monte carlo calculations performed on the GPU.    *
//                                                    *
// Author: Zachariah Bryant                           *
//*****************************************************


//**************
//   Headers   *
//**************
#include <iostream>
#include <fstream>
#include <string.h>

//Header that contains our ising model class that performs
//operations on the GPU
#include "lattice.cuh"

//Header for gnuplot
#include "gnuplot-iostream.h"

using namespace std;


//**************************************
//   Definition of all the variables   *
//**************************************
#define LATTSIZE 80 //Must be multiple of 8 for now
#define CONFIGS 1000
#define EQSWEEPS 300
#define JAY .9
#define MAGH 0.1
#define BETA 0.2
#define STARTTEMP 0.2
#define TEMPCHANGE 0.02
#define TEMPLIMIT 5



//Averages a vector
double Average(vector<double> avgspin)
{
  double total{0};

  for(unsigned int i = 0; i < avgspin.size(); i++)
  total += avgspin[i];

  return total/avgspin.size();
}

double Standard_Deviation(vector<double> avgspin)
{
  double x{0};
  double y{Average(avgspin)};

  for(int i = 0; i < CONFIGS; i++)
  x += pow((avgspin[i] - y), 2);

  x = sqrt(x)/sqrt(CONFIGS*(CONFIGS-1));
  return x;
}



//**********************
//    Main Function    *
//**********************
int main()
{

  //Create our Ising Model object for operating on
  ising_model ising(LATTSIZE, BETA, JAY, MAGH);

  //Defines files for logging data
  fstream File0, File1;

  //Thermalize lattice and log data
  File0.open("Spin_vs_Eq.dat", ios::out | ios::trunc );
  cout << "************************ THERMALIZING ************************ \n";
  for(int i = 0;  i < EQSWEEPS; i++)
  {
    File0 << i << " " << ising.AverageSpin() << "\n";
    File0.flush();

    ising.Equilibrate();
  }
  File0.close();



  //Iterate Temperature of the object and log the data
  vector<double> AvgSpin(CONFIGS);

  string name = "Spin_vs_T(J=";
  string Jay = to_string(JAY);
  Jay.resize(4); //Truncates J to 4 characters
  name += Jay;
  name += ").dat";
  File1.open( name, ios::out | ios::trunc );
  double average{0};
  double temp{STARTTEMP};
  ising.SetBeta(1/temp);

  //Calculation of avg. spin vs temperature, i.e. we change beta,
  cout << "****************** ITERATING TEMP ************************ \n";
  do
  {
    for(int i = 0; i < 10; i++)
    ising.Equilibrate();

    //Collects the average spin based on how many CONFIGS desired
    for(int i = 0; i < CONFIGS; i++)
    {
      //Seperates configurations for measurement
      for(int j = 0; j < 2; j++)
      ising.Equilibrate();

      //Measure avg. over all spins of a given configurations
      AvgSpin[i] = ising.AverageSpin();
    }

    average = Average(AvgSpin);
    File1 << temp << "  " << average << " " << Standard_Deviation(AvgSpin) <<  "\n";
    File1.flush();


    temp += TEMPCHANGE;
    ising.SetBeta(1/(temp));
  }
  while(temp < TEMPLIMIT);

  File1.close();

  //Plots the data automatically
  Gnuplot gp;
  gp << "set title \"Average Spin vs T\" \n";
  gp << "set xlabel \" Temperature (1/Beta)\"\n";
  gp << "set ylabel \"Average Spin\" \n";
  gp << "plot [0:5] '"<< name <<"' using 1:2:3 w yerr\n";
  gp << "set term png \n";
  gp << "set output \"" << name << "\".png \n";
  gp << "replot\n";
  gp << "set term x11 \n";


  cout << "****** FINISHED ******** \n \n" ;


  return 0;

}
