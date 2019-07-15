library(neuralnet)
# Binary classification
set.seed(1000)

nn <- neuralnet((Species == "setosa") ~ Petal.Length + Petal.Width, iris, 
                linear.output = FALSE,threshold=0.01,act.fct = "logistic")
print(nn$result.matrix) # the parameters
plot(nn)
# predict one new value to check the arithmetic
print(myp<-predict(nn,data.frame(Petal.Length=0.5,Petal.Width=0.85)))

## manually calculate from parameters (see plot)
x1<-(0.5*nn$result.matrix["Petal.Length.to.1layhid1",1]+
       0.85*nn$result.matrix["Petal.Width.to.1layhid1",1])+nn$result.matrix["Intercept.to.1layhid1",1]
y1<-1/(1+exp(-x1))
x2<- y1*nn$result.matrix['1layhid1.to.Species == "setosa"',1]+nn$result.matrix['Intercept.to.Species == "setosa"',1]
y2<-1/(1+exp(-x2))
cat("manual=",y2,"builtin=",myp,"\n");


##
softplus <- function(x) log(1 + exp(x))
nn <- neuralnet((Species == "setosa") ~ Petal.Length + Petal.Width, iris,
                linear.output = FALSE, hidden = c(3, 2), act.fct = softplus)