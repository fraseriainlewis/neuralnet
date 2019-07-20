#------------------------------------------------------------------------------------------------------------#
# create a simulated data set for regression
# 9 features, X1-X9, response y is a linear function of these, ynl1 and ynl2 as non-linear functions of y
# CSV export, with two files, features= "features.csv", 
# the different y responses are "labelsL1.csv", "labelsNL1.csv", "labelsNL2.csv"
#------------------------------------------------------------------------------------------------------------#
setwd("/Users/fraser/myrepos/neuralnet");
library(MASS)
set.seed(1000)
# create Xs as from multi-normal for ease, also each are indep of each other 
Sigma <- matrix(data=rep(0,9*9),ncol=9,byrow=TRUE)
for(i in 1:ncol(Sigma)){Sigma[i,i]<-1;} # diagonal covariance matrix
X<-mvrnorm(1000,mu=rep(0,ncol(Sigma)),Sigma=Sigma)

int<-2.0;# intercept
betas<-c(1.0,-2.5,3.0,0.5,-6,-0.9,20,-1.5,0.3) # coefficients in additive expression for mean

y<-rnorm(1000,mean=X%*%betas+int,sd=1)

# combined into a data.frame and create non-linear functions of y
dat<-data.frame(y,ynl1=log(1+exp(y)),ynl2=sin(y),X);
# standardize X variables to mean=0 sd=1 
for(i in 4:ncol(dat)){dat[,i]<-(dat[,i]-mean(dat[,i]))/sd(dat[,i]);}

summary(lm(y~1+X1+X2+X3+X4+X5+X6+X7+X8+X9,data=dat)) # linear response

summary(lm(ynl1~1+X1+X2+X3+X4+X5+X6+X7+X8+X9,data=dat)) # linear fit to non-linear response1

summary(lm(ynl2~1+X1+X2+X3+X4+X5+X6+X7+X8+X9,data=dat)) # linear fit to non-linear response1

## create CSV files
write.csv(dat[,c("X1","X2","X3","X4","X5","X6","X7","X8","X9")],file="features.csv",row.names=FALSE)
write.csv(dat[,"y"],file="labelsL1.csv",row.names=FALSE)
write.csv(dat[,"ynl1"],file="labelsNL1.csv",row.names=FALSE)
write.csv(dat[,"ynl2"],file="labelsNL2.csv",row.names=FALSE)
# note - then manually delete the header row in the csv files

