#include <iostream>
#include <armadillo>

class envDAG {
    
    // This constructor has optional arguments, meaning you can skip them (which will result in them being set to 0).
    public:
    
    envDAG(const std::string datafile, const double alpha_w = 30, const double alpha_m = 30);
    
    //void PrintData(void) const;// the const keyword after the parameter list tells the compiler that this method won't modify the actual object

    //void fitDAG(const arma::umat dag);

   
    void resetDAG(const arma::umat dag, const arma::ivec pos);
    void step(const unsigned int actidx);// action index in actions matrix
    bool hasCycle();
    void fitDAG(void);

private:
    double alpha_w, alpha_m;
    double lnscore = -std::numeric_limits<double>::max();//most negative number
    std::string datafile;
    arma::mat rwdata;
    arma::uword n;// variables
    arma::uword N;//data points
    arma::umat nu; // prior precision for each node 
    arma::mat mu0; // prior mu0 for each node 
    arma::mat b; // this is the regression coefs for each node by each node
    arma::mat T,R; //
    arma::umat dag0; // empty dag created in constructor
    arma::umat dag_cp;//copy of dag0
    arma::ivec pos0 = arma::zeros<arma::ivec>(2);// allocate memory here as dimension is fixed - this is x,y coordination on DAG board
    arma::ivec pos_cp;// copy of pos0
    arma::imat actions = arma::zeros<arma::imat>(15,2);// 15 actions - rows, each with 2 parts - cols
    arma::uvec isactive, isactive_scratch,incomingedges;
    arma::umat graph;//used as scratch in cycle
    bool invalidAction=false;//if this is true then the action introduced a cycle 

   void setT(void); // Compute T = precision matrix in Wishart prior.
   void setR(void);
   double pDln(const unsigned int l, const arma::uvec YY);
   double gammalnmult(const double l, const double xhalf);
   void get_numincomingedges();
   
   


};

