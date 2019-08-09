/** c++ test.cpp -o a.out -std=c++11 -lboost_serialization -larmadillo -lmlpack -fopenmp -Wall **/

#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>
#include <mlpack/methods/ann/init_rules/const_init.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <ensmallen.hpp>
#include <mlpack/prereqs.hpp>


using namespace mlpack;
using namespace mlpack::ann;
using namespace arma;
using namespace std;

int main()
{
// set some debugging flags 
unsigned int checkCSV=1;// print out first parts of features and labels to check these imported correctly
unsigned int checkPredict=1;// print out parts of log probabilites - the model output

// set the random number seed - e.g. shuffling or random starting points in optim
arma::arma_rng::set_seed(100);
//arma::arma_rng::set_seed_random();

/**************************************************************************************************/
/**************************************************************************************************/
/** Load the training set - separate files for features and lables **/
/** note - data is read into matrix in column major, e.g. each new data point is a column - opposite from data file **/
arma::mat trainData, trainLabels01;
unsigned int i,j;
data::Load("s_features8.csv", trainData, true);// last arg is transpose - needed as boost is col major
data::Load("s_labelsB.csv", trainLabels01, true);// 

const arma::mat trainLabels = trainLabels01.row(0) + 1;// add +1 to all response values, mapping from 0-1 to 1-2
                                                 // NegativeLogLikelihood needs 1...number of classes

if(checkCSV){
	std::cout<<"#-------------------------------------------------------------------------#"<<std::endl;
	std::cout<<"#--------- Import CSV debugging on ---------------------------------------#"<<std::endl;
	std::cout<<"#-------------------------------------------------------------------------#"<<std::endl<<std::endl;
 	// print out to check the data is read in correctly
 	std::cout<<"\t1. Data dimensions check"<<std::endl;
 	arma::cout << "\tn rows="<<trainData.n_rows <<" n cols="<< trainData.n_cols << arma::endl;
 	arma::cout << "\tn rows="<< trainLabels.n_rows <<" n cols"<< trainLabels.n_cols << arma::endl;

 	std::cout<<std::endl<<"\t2. First 10 Labels"<<std::endl;
 	for(i=0;i<10;i++){
  					arma::cout <<"\t"<< trainLabels(0,i) << arma::endl;
 		}

 	std::cout<<std::endl<<"\t3. Features for first 2 observations"<<std::endl;
 	for(j=0;j<2;j++){ std::cout<<"\t--> obs number: "<<j<<std::endl;
 			for(i=0;i<trainData.n_rows;i++){
  								arma::cout <<"\t"<<trainData(i,j) << arma::endl;
 											}
 					}
 	std::cout<<"#-------------------------------------------------------------------------#"<<std::endl;
	std::cout<<"#--------- Import CSV debugging end --------------------------------------#"<<std::endl;
	std::cout<<"#-------------------------------------------------------------------------#"<<std::endl<<std::endl;

	} //end of if

/**************************************************************************************************/
/**************************************************************************************************/
/**************************************************************************************************/
/** MODELS - simple linear regression  **/

// PART1 - starting weights initialize to constant
std::cout<<"----- PART 1 ------"<<std::endl;
// initialise weights to a constant and use MSE as the loss function
FFN<NegativeLogLikelihood<>,RandomInitialization> model1(NegativeLogLikelihood<>(),RandomInitialization(-1,1));//default
// build layers
const size_t inputSize=trainData.n_rows;// 9 
//const size_t outputSize=1;
const size_t hiddenLayerSize=2;

model1.Add<Linear<> >(inputSize, hiddenLayerSize);
//model1.Add<SigmoidLayer<> >();
model1.Add<LogSoftMax<> >();

//model1.Add<IdentityLayer<> >();// needed = final output value is sum of weights and data

// set up optimizer 
ens::Adam optBAD(0.01, 2211, 0.9, 0.999, 1e-8, 0, 1e-4,false,true); //https://ensmallen.org/docs.html#rmsprop.
                 // 2nd arg is batch size,
                 // 5th arg is max iterations (0=no limit) 
                 // 6th is tolerance 
                 // 7th shuffle = false
                 // 8th clean re-start policy = true

ens::RMSProp opt(0.01, 32/*trainData.n_cols*/, 0.99, 1e-8, 0,1e-5,false,true); //https://ensmallen.org/docs.html#rmsprop.
ens::RMSProp optDF(0.01, 2211/*trainData.n_cols*/, 0.99, 1e-8, 0,1e-5,false,true);
                 // 2nd arg is batch size,
                 // 5th arg is max iterations (0=no limit) 
                 // 6th is tolerance 
                 // 7th shuffle = false
                 // 8th clean re-start policy = true



arma::cout<<"-------empty params------------------------"<<arma::endl;//empty params as not yet allocated
arma::cout << model1.Parameters() << arma::endl;
double lossAuto=model1.Train(trainData, trainLabels,optBAD);
arma::cout<<"-------final params------------------------"<<arma::endl;
arma::cout << model1.Parameters() << arma::endl;

// Use the Predict method to get the assignments.
arma::mat assignments;
model1.Predict(trainData, assignments);


if(checkPredict){
  std::cout<<"#-------------------------------------------------------------------------#"<<std::endl;
  std::cout<<"#--------- log probabilities debugging on --------------------------------#"<<std::endl;
  std::cout<<"#-------------------------------------------------------------------------#"<<std::endl<<std::endl;

  std::cout<<"Predictions shape rows  : "<<assignments.n_rows<<std::endl;
  std::cout<<"Predictions shape cols : "<<assignments.n_cols<<std::endl;
  std::cout<<std::endl<<"\t1. selected output records, first 10 records"<<std::endl;

  for(i=0;i<10;i++){//assignments.n_cols;i++){
       cout<<"\t"<<i<<"\t["<<assignments(0,i)<<","<<assignments(1,i)<<"]"<<endl;
   }
   std::cout<<std::endl<<"\t2. selected output records, first 10 records which correspond to true label"<<std::endl;
   for(i=0;i<10;i++){//assignments.n_cols;i++){
       cout<<"\t"<<i<<"\t["<<assignments(trainLabels(0,i)-1,i)<<"]"<<endl;
   }

   std::cout<<std::endl<<"\t3. selected output records, last 10 records"<<std::endl;
   for(i=assignments.n_cols-10;i<assignments.n_cols;i++){//assignments.n_cols;i++){
       cout<<"\t"<<i<<"\t["<<assignments(0,i)<<","<<assignments(1,i)<<"]"<<endl;
   }
  std::cout<<std::endl<<"\t4. selected output records, last 10 records which correspond to true label"<<std::endl;
  for(i=assignments.n_cols-10;i<assignments.n_cols;i++){//assignments.n_cols;i++){
       cout<<"\t"<<i<<"\t["<<assignments(trainLabels(0,i)-1,i)<<"]"<<endl;
   }

} // end of prob debug check

// compute the negative log like loss manually 
double lossManual=0;

for(i=0;i<assignments.n_cols;i++){
       lossManual+= -(assignments(trainLabels(0,i)-1,i));
}

std::cout<<"NLL (from Train()) on full data set="<<lossAuto<<std::endl;
std::cout<<"NLL manual - and correct - on full data set="<<lossManual<<std::endl;
// these are identical if batch size is set to full data set size - or change to false to true for randomize at each batch

}