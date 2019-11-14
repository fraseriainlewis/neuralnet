/** To build on mac use following line at terminal in the folder with the .c file. Note - need gnu GSL installed (tarball source install works fine)
and needs Xcode installed for clang

/Applications/MATLAB_R2019b.app/bin/mex -I/usr/local/include -L/usr/local/lib -lgsl -lgslcblas dagCombinations.c -R2018a

  **/


#include "mex.h"
#include "math.h"
#include <gsl/gsl_combination.h>
/* The computational routine */
/* this computes n choose r, for n=numnodes and r from 0 to numnodes, so gives total number of 
 * possible parent combinations per node. To get total DAGs (ignoring cycles) this is just the
 * paraent combination raised to power of numnodes */
void dagCombinations(unsigned int numNodes, unsigned int numDAGs, double *out)
{


    
    mexPrintf("here %u\n",numDAGs);
    //tablesize[0]=pow(total,numnodes);
    
}



/* The gateway function */
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{
    unsigned int nnodes;               /* dimension of DAG */
    unsigned int ndags;   /** number of combinations - including duplicates due to orders */
    double *outMatrix_ptr;              /* output matrix */

    /* check for proper number of arguments */
    if(nrhs!=2) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs","two inputs required.");
    }
    if(nlhs!=1) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nlhs","One output required.");
    }

    /* make sure the first input argument is type uint32 */
    if( ! (mxIsUint32(prhs[0]))) 
         {
        mexErrMsgIdAndTxt("MyToolbox:cycle:notUint32","All inputs must be type uint32.");
    }
    
    
    /* get the value of the scalar input  */
    nnodes = (unsigned int)mxGetScalar(prhs[0]);
    ndags = (unsigned int)mxGetScalar(prhs[1]);
    
    /** need to compute the number of output rows */


    /* create the output matrix - using 1x1 - could use scalar instead */
    plhs[0] = mxCreateDoubleMatrix((mwSize)ndags,(mwSize)nnodes,mxREAL);

    /* get a pointer to the real data in the output matrix */
    #if MX_HAS_INTERLEAVED_COMPLEX
    outMatrix_ptr = mxGetDoubles(plhs[0]);
    #else
    outMatrix_ptr = mxGetPr(plhs[0]);
    #endif

    /* call the computational routine */
    dagCombinations(nnodes,ndags,outMatrix_ptr);

}




