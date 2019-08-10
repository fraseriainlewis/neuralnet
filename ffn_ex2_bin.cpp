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



/***********************************************************************************************************/
/***********************************************************************************************************/
/** MAIN                                                                                                   */
/***********************************************************************************************************/
/***********************************************************************************************************/

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
arma::mat featureData, labels01;
unsigned int i,j;
data::Load("features.csv", featureData, true);// last arg is transpose - needed as boost is col major
data::Load("labelsBNL1.csv", labels01, true);// 

const arma::mat labels = labels01.row(0) + 1;// add +1 to all response values, mapping from 0-1 to 1-2
                                                 // NegativeLogLikelihood needs 1...number of classes                                            

if(checkCSV){
	std::cout<<"#-------------------------------------------------------------------------#"<<std::endl;
	std::cout<<"#--------- Import CSV debugging on ---------------------------------------#"<<std::endl;
	std::cout<<"#-------------------------------------------------------------------------#"<<std::endl<<std::endl;
 	// print out to check the data is read in correctly
 	std::cout<<"\t1. Data dimensions check"<<std::endl;
 	arma::cout << "\tn rows="<<featureData.n_rows <<" n cols="<< featureData.n_cols << arma::endl;
 	arma::cout << "\tn rows="<< labels.n_rows <<" n cols"<< labels.n_cols << arma::endl;

 	std::cout<<std::endl<<"\t2. First 10 Labels"<<std::endl;
 	for(i=0;i<10;i++){
  					arma::cout <<"\t"<< labels(0,i) << arma::endl;
 		}

 	std::cout<<std::endl<<"\t3. Features for first 2 observations"<<std::endl;
 	for(j=0;j<2;j++){ std::cout<<"\t--> obs number: "<<j<<std::endl;
 			for(i=0;i<featureData.n_rows;i++){
  								arma::cout <<"\t"<<featureData(i,j) << arma::endl;
 											}
 					}
 	std::cout<<"#-------------------------------------------------------------------------#"<<std::endl;
	std::cout<<"#--------- Import CSV debugging end --------------------------------------#"<<std::endl;
	std::cout<<"#-------------------------------------------------------------------------#"<<std::endl<<std::endl;

	} //end of if

/**************************************************************************************************/
/**************************************************************************************************/
/**************************************************************************************************/
/** MODEL DEFINITION  **/

// PART1 - starting weights initialize to constant
std::cout<<"----- PART 1 ------"<<std::endl;
// initialise weights default random and use Neg Log Lik as the loss function
FFN<NegativeLogLikelihood<>,RandomInitialization> model1(NegativeLogLikelihood<>(),RandomInitialization(-1,1));//default

// build layers
const size_t inputSize=featureData.n_rows;// e.g. 8 
//const size_t outputSize=1;
const size_t hiddenLayerSize=2;

model1.Add<Linear<> >(inputSize, hiddenLayerSize);
model1.Add<LogSoftMax<> >();

/**************************************************************************************************/
/************************END MODEL DEFN************************************************************/
/**************************************************************************************************/
 
/**************************************************************************************************/
/********************      Define Optimizer                                                 *******/
/**************************************************************************************************/
// set up optimizer 
//ens::RMSProp opt(0.01, featureData.n_cols, 0.99, 1e-8, 1000000,1e-5,false,true);//batch size is all obs
                 // 2nd arg is batch size,
                 // 5th arg is max iterations (0=no limit) 
                 // 6th is tolerance 
                 // 7th shuffle = false
                 // 8th clean re-start policy = true

ens::Adam opt(0.01, featureData.n_cols, 0.9, 0.999, 1e-8, 0, 1e-5,false,true); //https://ensmallen.org/docs.html#rmsprop.

double lossAuto=1e+300;
arma::cout<<"-------empty params------------------------"<<arma::endl;//empty params as not yet allocated
arma::cout << model1.Parameters() << arma::endl;
model1.Train(featureData, labels,opt);
lossAuto=model1.Evaluate(featureData, labels);
arma::cout<<"-------final params------------------------"<<arma::endl;
arma::cout << model1.Parameters() << arma::endl;

// Use the Predict method to get the assignments.
arma::mat assignments;
model1.Predict(featureData, assignments);

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
       cout<<"\t"<<i<<"\t["<<assignments(labels(0,i)-1,i)<<"]"<<endl;
   }

   std::cout<<std::endl<<"\t3. selected output records, last 10 records"<<std::endl;
   for(i=assignments.n_cols-10;i<assignments.n_cols;i++){//assignments.n_cols;i++){
       cout<<"\t"<<i<<"\t["<<assignments(0,i)<<","<<assignments(1,i)<<"]"<<endl;
   }
  std::cout<<std::endl<<"\t4. selected output records, last 10 records which correspond to true label"<<std::endl;
  for(i=assignments.n_cols-10;i<assignments.n_cols;i++){//assignments.n_cols;i++){
       cout<<"\t"<<i<<"\t["<<assignments(labels(0,i)-1,i)<<"]"<<endl;
   }

} // end of prob debug check

// compute the negative log like loss manually 
double lossManual=0;
double predp2,predp1,sen,spec,acc;
uword P,N,TP,TN;
P=0;
N=0;
TP=0;
TN=0;
for(i=0;i<assignments.n_cols;i++){
       lossManual+= -(assignments(labels(0,i)-1,i));

       predp2=assignments(1,i);//log prob of being in class 2
       predp1=assignments(0,i);//log prob of being in class 1
       
       if(labels(0,i)==2){//truth is class 2 - positives
         P++;//count number of class 2
         if(predp2>=predp1){//truth is class 2 and predict class 2
                                           TP++;//increment count of true positives TP
            }
         } //end of count positives
       if(labels(0,i)==1){//truth is class 1
         N++;// count number of class 1 - negatives
         if(predp1>=predp2){//truth is class 1 and predict class 1
                                           TN++;//increment count of true negative FN
       }
       }
                
}
sen=(double)TP/(double)P;
spec=(double)TN/(double)N;
acc=((double)TP+(double)TN)/((double)P+(double)N);//overall accuracy
  
std::cout<<"NLL (from Evaluate()) on full data set="<<lossAuto<<std::endl;
std::cout<<"NLL manual - and correct - on full data set="<<lossManual<<std::endl;
std::cout<<"P= "<<P<<"  TP="<<TP<<"  N= "<<N<<"  TN= "<<TN<<std::endl;
std::cout<<"sensitivity = "<<std::setprecision(5)<<fixed<<sen<<" specificity = "<<spec<<" accuracy= "<<acc<<std::endl;



}