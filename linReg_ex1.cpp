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

// set the random number seed - e.g. shuffling or random starting points in optim
arma::arma_rng::set_seed(100);
//arma::arma_rng::set_seed_random();

/**************************************************************************************************/
/** Load the training set - separate files for features and lables (regression) **/
/** note - data is read into matrix in column major, e.g. each new data point is a column - opposite from data file **/
arma::mat trainData, trainLabels;
int i=0;
data::Load("trainfeatures.csv", trainData, true);
data::Load("trainlabels.csv", trainLabels, true);// regression response

// print out to check the data is read in correctly
arma::cout << "n rows="<<trainData.n_rows <<"n cols="<< trainData.n_cols << arma::endl;
arma::cout << "n rows="<< trainLabels.n_rows <<"n cols"<< trainLabels.n_cols << arma::endl;

std::cout<<"First 10 Labels:"<<std::endl;
for(i=0;i<10;i++){
 arma::cout << trainLabels(0,i) << arma::endl;
}

int j=0;
std::cout<<"Features for observation: "<<j<<std::endl;
for(i=0;i<trainData.n_rows;i++){
 arma::cout << trainData(i,j) << arma::endl;
}
j=1;
std::cout<<"Features for observation:"<<j<<std::endl;
for(i=0;i<trainData.n_rows;i++){
 arma::cout << trainData(i,1) << arma::endl;
}
/**************************************************************************************************/
/** MODELS - simple linear regression examples using different optim and start criteria **/

// PART1 - starting weights initialize to constant
std::cout<<"----- PART 1 ------"<<std::endl;
// initialise weights to a constant and use MSE as the loss function
FFN<MeanSquaredError<>,ConstInitialization> model1(MeanSquaredError<>(),ConstInitialization(0.9));
// build layers
model1.Add<Linear<> >(trainData.n_rows,1);
model1.Add<IdentityLayer<> >();// needed = final output value is sum of weights and data
// set up optimizer 
ens::RMSProp opt(0.01, 1060, 0.99, 1e-8, 0,1e-8,false,true); //https://ensmallen.org/docs.html#rmsprop.
                 // 2nd arg is batch size,
                 // 5th arg is max iterations (0=no limit) 
                 // 6th is tolerance 
                 // 7th shuffle = false
                 // 8th clean re-start policy = true

arma::cout<<"-------empty params------------------------"<<arma::endl;//empty params as not yet allocated
arma::cout << model1.Parameters() << arma::endl;
model1.Train(trainData, trainLabels,opt);
arma::cout<<"-------final params------------------------"<<arma::endl;
arma::cout << model1.Parameters() << arma::endl;

model1.ResetParameters();// reset parameters to their initial values - should be used with clean re-start policy = true
                         // to continue with optim from previous solution use clean re-start policy = false and do not reset
arma::cout<<"-------re-start params------------------------"<<arma::endl;
arma::cout << model1.Parameters() << arma::endl;
model1.Train(trainData, trainLabels,opt);
arma::cout<<"-------re-start final params------------------------"<<arma::endl;
arma::cout << model1.Parameters() << arma::endl;

// PART 2 - manually starting set weights
std::cout<<"----- PART 2 ------"<<std::endl;
// initialise weights to a constant and use MSE as the loss function
FFN<MeanSquaredError<>,ConstInitialization> model2(MeanSquaredError<>(),ConstInitialization(0.9));
// build layers
Linear<>* linearModule = new Linear<>(trainData.n_rows,1);
model2.Add(linearModule);
model2.Add<IdentityLayer<> >();

model2.Evaluate(trainData, trainLabels);
arma::mat inits(trainData.n_rows+1,1);
for(i=0;i<10;i++){inits(i,0)=0.05;}

linearModule->Parameters() = std::move(inits);

arma::cout<<"-------manual set init params------------------------"<<arma::endl;//empty params as not yet allocated
arma::cout << model2.Parameters() << arma::endl;

// set up optimizer 
ens::RMSProp opt2(0.01, 1060, 0.99, 1e-8, 0,1e-8,false,true); //https://ensmallen.org/docs.html#rmsprop.
                 // 2nd arg is batch size,
                 // 5th arg is max iterations (0=no limit) 
                 // 6th is tolerance 
                 // 7th shuffle = false
                 // 8th clean re-start policy = true


model2.Train(trainData, trainLabels,opt2);
arma::cout<<"-------final params------------------------"<<arma::endl;
arma::cout << model2.Parameters() << arma::endl;

// PART 3 - random starting weights
std::cout<<"----- PART 3 ------"<<std::endl;
// initialise weights to a constant and use MSE as the loss function
FFN<MeanSquaredError<>,RandomInitialization> model3(MeanSquaredError<>(),RandomInitialization(-1.0,1.0));
// build layers
model3.Add<Linear<> >(trainData.n_rows,1);
model3.Add<IdentityLayer<> >();// needed = final output value is sum of weights and data
// set up optimizer 
ens::RMSProp opt3(0.01, 1060, 0.99, 1e-8, 10000,1e-8,false,true); //https://ensmallen.org/docs.html#rmsprop.
                 // 2nd arg is batch size,
                 // 5th arg is max iterations (0=no limit) 
                 // 6th is tolerance 
                 // 7th shuffle = false
                 // 8th clean re-start policy = true

arma::cout<<"-------empty params------------------------"<<arma::endl;//empty params as not yet allocated
arma::cout << model3.Parameters() << arma::endl;
model3.Train(trainData, trainLabels,opt3);
arma::cout<<"-------final params------------------------"<<arma::endl;
arma::cout << model3.Parameters() << arma::endl;

arma::arma_rng::set_seed(100);
model3.ResetParameters();// reset parameters to their initial values - should be used with clean re-start policy = true
                         // to continue with optim from previous solution use clean re-start policy = false and do not reset
arma::cout<<"-------re-start params------------------------"<<arma::endl;
arma::cout << model3.Parameters() << arma::endl;
model3.Train(trainData, trainLabels,opt3);
arma::cout<<"-------re-start final params------------------------"<<arma::endl;
arma::cout << model3.Parameters() << arma::endl;

arma::arma_rng::set_seed(100);
model3.ResetParameters();// reset parameters to their initial values - should be used with clean re-start policy = true
                         // to continue with optim from previous solution use clean re-start policy = false and do not reset
arma::cout<<"-------re-start params------------------------"<<arma::endl;
arma::cout << model3.Parameters() << arma::endl;
model3.Train(trainData, trainLabels,opt3);
arma::cout<<"-------re-start final params------------------------"<<arma::endl;
arma::cout << model3.Parameters() << arma::endl;


}
