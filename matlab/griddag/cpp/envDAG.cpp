// envDAG.cpp
#include "envDAG.hpp"
#include <iostream>
#include <armadillo>

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
    b = arma::zeros<arma::mat>(n,n);


}

void envDAG::PrintData(void) const {
	arma::cout<<"number of variables="<<rwdata.n_cols<<arma::endl;
}

double envDAG::getAlpha_w(void) const {
    
    std::cout<<"_w="<<alpha_w<<" _m="<<alpha_m<<std::endl;
    
    return alpha_w;
}

