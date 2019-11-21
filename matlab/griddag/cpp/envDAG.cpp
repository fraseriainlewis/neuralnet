// envDAG.cpp
#include "envDAG.hpp"
#include <iostream>
#include <armadillo>

#define DEBUG

envDAG::envDAG(const std::string datafile, const double _alpha_w, const double _alpha_m): alpha_w(_alpha_w), alpha_m(_alpha_m) { 
	std::cout<<"constructing!"<<std::endl;

    /** firstly read in the raw data from disk - CSV without header line **/
	bool ok = rwdata.load(datafile);

    if(ok == false){
    	std::cout << "problem with loading raw data from file" << std::endl;
    	exit(-1);
    }

    /** compute prior constants used across all model fitting */
    // nu  = ones(1,n);  % prior precision for each node = 1
    // mu0 = zeros(1,n); % prior mu0 for each node = 0
    // b=zeros(n,n);     % this is the regression coefs - all zero 

    n = rwdata.n_cols; //number of variables - dimension of DAG
    std::cout<<"n="<<n<<std::endl;
    nu = arma::ones<arma::umat>(1,n);
    mu0 = arma::zeros<arma::mat>(1,n);
    mu0(0,0) = 0.1;
    mu0(0,1) = -0.3;
    mu0(0,2) = 0.2;

    b = arma::zeros<arma::mat>(n,n);
    b(0,2)=1;
    b(1,2)=1;               
   

    // % Compute T = precision matrix in Wishart prior.
    // %%%%%%% equation 5 and 6 in Geiger and Heckerman
    sigmainv=priorPrec(nu,b);
    arma::cout<<"priorPrec\n"<<sigmainv<<arma::endl;
    

}

void envDAG::PrintData(void) const {
	arma::cout<<"number of variables="<<rwdata.n_cols<<arma::endl;
}

double envDAG::getAlpha_w(void) const {
    
    std::cout<<"_w="<<alpha_w<<" _m="<<alpha_m<<std::endl;
    
    return alpha_w;
}


arma::mat envDAG::priorPrec(const arma::umat nu, const arma::mat b){

 arma::uword n=3;//b.n_rows; // number of variables
 arma::uword wsize=1;
 
 arma::mat w1  = arma::zeros<arma::mat>(wsize,wsize);
 arma::mat w2,e1,sigmainv;
 unsigned int i=0;

 w1(0,0) = 1/nu(0,0);
 // w1 is current matrix
 // w2 is new matrix

#ifdef DEBUG
 for(i=1;i<n;i++){
  	wsize=i;
  	w2  = arma::zeros<arma::mat>(wsize+1,wsize+1); // initialize
  	e1=b.submat(0,i,(i-1),i);// row 0:(i-1), col i 
    //e1=e2;
  	w2.submat(0,0,(wsize-1),(wsize-1)) = w1+kron(e1,e1.t())/nu(0,i); // set submatrix top corner
  	w2.submat(wsize,0,wsize,(wsize-1))= -e1.t()/nu(0,i); //last row
  	w2.submat(0,wsize,(wsize-1),wsize)= -e1/nu(0,i);  // last col
  	w2(wsize,wsize) = 1/nu(0,i); // bottom right cell

  	w1=w2; // update copy new matrix to current matrix
 }
#endif
 
 //m = w2; // return the final matrix Wn
 
 sigmainv=w2;
 double sigmaFactor = (alpha_m+1)/(alpha_m*(alpha_w-n-1));
 arma::mat T =inv(sigmainv)/sigmaFactor;

 return(T);






}

