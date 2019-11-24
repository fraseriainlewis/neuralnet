#include "envDAG.hpp"
#include <armadillo>
#include <iostream>
 
int main()
{
  
  // example of testing for success
 std::string datafile = "n50m1000.csv";// "test3.csv"; 
unsigned int i;

std::cout << "Hello World!" << std::endl;

envDAG env1(datafile, 100,100);

 /* auto myArea = env1.getAlpha_w();
  std::cout<<env1.getAlpha_w()<<std::endl;
  std::cout<<myArea<<std::endl;
*/

 // env1.PrintData();
 // env1.fitDAG();
#ifdef A
dag0<-matrix(data=rep(0,n*n),ncol=n);
dag0[1,1:4]=c(0,1,1,1)
dag0[2,1:4]=c(0,0,0,0)
dag0[3,1:4]=c(0,1,0,0)
dag0[4,1:4]=c(0,0,1,0)
dag0[20,42]=1;
dag0[42,1:20]=1;

#endif

arma::umat daga = arma::zeros<arma::umat>(50,50);

/*daga(1-1,2-1)=1;daga(1-1,3-1)=1;
daga(2-1,3-1)=1;daga(2-1,4-1)=1;
daga(3-1,4-1)=1;
*/

/*daga(0,0)=0;daga(0,1)=1;daga(0,2)=1;daga(0,3)=1;
daga(1,0)=0;daga(1,1)=0;daga(1,2)=0;daga(1,3)=0;
daga(2,0)=0;daga(2,1)=1;daga(2,2)=0;daga(2,3)=0;
daga(3,0)=0;daga(3,1)=0;daga(3,2)=1;daga(3,3)=0;
*/   
//daga(19,41)=1;
//for(i=0;i<20;i++){daga(41,i)=1;}

//arma::cout<<"daga="<<arma::endl<<daga<<arma::endl;

  env1.fitDAG(daga);

/*arma::mat A(5,5,arma::fill::randu);

double val;
double sign;

log_det(val, sign, A);
arma::cout<<"val="<<val<<" sign="<<sign<<" logdet="<<det(A)<<arma::endl<<A;
*/



  return 0;
}
