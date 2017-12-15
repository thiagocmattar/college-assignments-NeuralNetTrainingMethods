rm(list=ls())

library(RSNNS)
library(NeuralNetTools)

library(readxl)
eneff <- read_excel("~/RNA/TP FINAL/Data1 - Energy/ENB2012_data.xlsx")
labs<-c('Relative Compactness','Surface Area','Wall Area','Roof Area',
       'Overall Height','Orientation','Glazing Area','Glazing Area Distribution',
       'Heating Load','Cooling Load')


e2y1<-matrix(0,nrow=100,ncol=3)
e2y2<-matrix(0,nrow=100,ncol=3)
e2y1.mean<-matrix(0,35,3)
e2y2.mean<-matrix(0,35,3)
e2y1.sd<-matrix(0,35,3)
e2y2.sd<-matrix(0,35,3)
k<-1

for(j in 5:28)
{
  print(c("ITERAÇÃO EXTERNA NUMERO",j))
  print(Sys.time())
for(i in 1:20)
{
  #Embaralhando o dataset para remoção de viés
  N<-nrow(eneff)
  set.seed(768)
  eneff<-eneff[sample(N,N),]
  
  #Separando em treino e teste
  inputs<-as.matrix(eneff[,1:8])
  outputs<-as.matrix(eneff[,9:10])
  dados<- splitForTrainingAndTest(inputs,outputs,ratio = 0.3)
  dados<- normTrainingAndTestSet(dados)

  rede1<-mlp(dados$inputsTrain, dados$targetsTrain, size = c(j), maxit = 100,
             initFunc = "Randomize_Weights", initFuncParams = c(-0.3, 0.3),
             learnFunc = "Rprop", learnFuncParams = c(0.1,0.1),
             updateFunc = "Topological_Order", updateFuncParams = 0.0,
             hiddenActFunc = "Act_Logistic",shufflePatterns = TRUE, 
             linOut = TRUE, inputsTest = NULL, targetsTest = NULL)
  
  plotIterativeError(rede1) 
  pred1<-predict(rede1,dados$inputsTest)
  e2y1[i,1]<-sum((pred1[,1]-dados$targetsTest[,1])^2)/nrow(dados$targetsTest)
  e2y2[i,1]<-sum((pred1[,2]-dados$targetsTest[,2])^2)/nrow(dados$targetsTest)
  
  rede2<-mlp(dados$inputsTrain, dados$targetsTrain, size = c(j), maxit = 100,
             initFunc = "Randomize_Weights", initFuncParams = c(-0.3,0, 0.3),
             learnFunc = "BackpropWeightDecay", learnFuncParams = 0.1,
             updateFunc = "Topological_Order", updateFuncParams = 0.0,
             hiddenActFunc = "Act_Logistic",shufflePatterns = TRUE, 
             linOut = TRUE, inputsTest = NULL, targetsTest = NULL)
  
  plotIterativeError(rede2) 
  pred2<-predict(rede2,dados$inputsTest)
  e2y1[i,2]<-sum((pred2[,1]-dados$targetsTest[,1])^2)/nrow(dados$targetsTest)
  e2y2[i,2]<-sum((pred2[,2]-dados$targetsTest[,2])^2)/nrow(dados$targetsTest)
  
  rede3<-mlp(dados$inputsTrain, dados$targetsTrain, size = c(j), maxit = 100,
             initFunc = "Randomize_Weights", initFuncParams = c(-0.3, 0, 0.3),
             learnFunc = "SCG", learnFuncParams = 0.1,
             updateFunc = "Topological_Order", updateFuncParams = 0.0,
             hiddenActFunc = "Act_Logistic",shufflePatterns = TRUE, 
             linOut = TRUE, inputsTest = NULL, targetsTest = NULL)
  
  plotIterativeError(rede3) 
  pred3<-predict(rede3,dados$inputsTest)
  e2y1[i,3]<-sum((pred3[,1]-dados$targetsTest[,1])^2)/nrow(dados$targetsTest)
  e2y2[i,3]<-sum((pred3[,2]-dados$targetsTest[,2])^2)/nrow(dados$targetsTest)
  
}

e2y1.mean[k,]<-c(mean(e2y1[,1]),mean(e2y1[,2]),mean(e2y1[,3]))
e2y1.sd[k,]<-c(sd(e2y1[,1]),sd(e2y1[,2]),sd(e2y1[,3]))
e2y2.mean[k,]<-c(mean(e2y2[,1]),mean(e2y2[,2]),mean(e2y2[,3]))
e2y2.sd[k,]<-c(sd(e2y2[,1]),sd(e2y2[,2]),sd(e2y2[,3]))
k<-k+1
}

#Y1: Erro quadrático médio x #neuronios na camada escondida
plot(4:27,e2y1.mean[1:24,1],type='l',ylim=c(0,1),ylab='',xlab='')
par(new=T)
plot(4:27,e2y1.mean[1:24,2],type='l',col='red',ylim=c(0,1),ylab='',xlab='')
par(new=T)
plot(4:27,e2y1.mean[1:24,3],type='l',col='blue',ylim=c(0,1),
     ylab='e² médio (Y1)',xlab='nº neurons hidden layer')

#Y1: Desvio padrão x #neurônios na camada escondida
plot(4:27,e2y1.sd[1:24,1],type='l',ylim=c(0,5),ylab='',xlab='')
par(new=T)
plot(4:27,e2y1.sd[1:24,2],type='l',col='red',ylim=c(0,5),ylab='',xlab='')
par(new=T)
plot(4:27,e2y1.sd[1:24,3],type='l',col='blue',ylim=c(0,5),
     ylab='STD (Y1)',xlab='nº neurons hidden layer')

#Y2: Erro quadrático médio x #neuronios na camada escondida
plot(4:27,e2y2.mean[1:24,1],type='l',ylim=c(0.5,2),ylab='',xlab='')
par(new=T)
plot(4:27,e2y2.mean[1:24,2],type='l',col='red',ylim=c(0.5,2),ylab='',xlab='')
par(new=T)
plot(4:27,e2y2.mean[1:24,3],type='l',col='blue',ylim=c(0.5,2),
     ylab='e² médio (Y2)',xlab='nº neurons hidden layer')

#Y2: Desvio padrão x #neurônios na camada escondida
plot(4:27,e2y2.sd[1:24,1],type='l',ylim=c(0.5,5),ylab='',xlab='')
par(new=T)
plot(4:27,e2y2.sd[1:24,2],type='l',col='red',ylim=c(0.5,5),ylab='',xlab='')
par(new=T)
plot(4:27,e2y2.sd[1:24,3],type='l',col='blue',ylim=c(0.5,5),
     ylab='STD (Y2)',xlab='nº neurons hidden layer')

residuos1 <- c(pred3[,1]-dados$targetsTest[,1])
residuos1 <- c(pred1[,1]-dados$targetsTest[,1])

acf(residuos,lag.max = 1000, type = c("correlation"),plot = TRUE, na.action = na.fail)
ccf(residuos,dados$targetsTest[,2],lag.max = 1000, type = c("correlation"),plot = TRUE, na.action = na.fail)
