#include <iostream>
#include <armadillo>

class envDAG {
    
    // This constructor has optional arguments, meaning you can skip them (which will result in them being set to 0).
    public:
    
    envDAG(const std::string datafile, const double alpha_w = 30, const double alpha_m = 30);
   
    double getAlpha_w(void) const; // the const keyword after the parameter list tells the compiler that this method won't modify the actual object
    
    void PrintData(void) const;

    void fitDAG(const arma::umat dag);

    void fitDAG(void);

private:
    double alpha_w, alpha_m;
    double lnscore = -std::numeric_limits<double>::max();
    std::string datafile;
    arma::mat rwdata;
    arma::uword n;// variables
    arma::uword N;//data points
    arma::umat nu; // prior precision for each node 
    arma::mat mu0; // prior mu0 for each node 
    arma::mat b; // this is the regression coefs for each node by each node
    arma::mat T,R; //
    arma::umat dag0; // empty dag created in constructor

   void setT(void); // Compute T = precision matrix in Wishart prior.
   void setR(void);
   double pDln(const unsigned int l, const arma::uvec YY);
   double gammalnmult(const double l, const double xhalf);

   


};

