#include <iostream>
#include <armadillo>

class envDAG {
    
    // This constructor has optional arguments, meaning you can skip them (which will result in them being set to 0).
    public:
    
    envDAG(const std::string datafile, const double alpha_w = 30, const double alpha_m = 30);
   

    double getAlpha_w(void) const; // the const keyword after the parameter list tells the compiler that this method won't modify the actual object
    
    void PrintData(void) const;

private:
    double alpha_w, alpha_m;
    std::string datafile;
    arma::mat rwdata;
    unsigned int n;
    arma::umat nu; // prior precision for each node 
    arma::mat mu0; // prior mu0 for each node 
    arma::mat b; // this is the regression coefs for each node by each node
    


};

