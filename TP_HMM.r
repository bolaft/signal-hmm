library("HMM")

# set working dir
setwd("/home/soufian/workspace/r/")

################ FUNCTIONS ################

HMMs_source <- function() {
  states <- c("s1", "s2")
  obs <- c("a", "b", "c")
  
  P1_source <- c(1, 0)
  A1_source <- matrix(c(0.1,0.9,0.6,0.4), nrow=2, ncol=2, byrow=T)
  B1_source <- matrix(c(0.1,0.3,0.6,0.4,0.2,0.4), nrow=2, ncol=3, byrow=T)
  
  lambda1_source <- initHMM(states, obs, P1_source, A1_source, B1_source)
  
  P2_source <- c(1, 0)
  A2_source <- matrix(c(0.4,0.6,0.8,0.2), nrow=2, ncol=2, byrow=T)
  B2_source <- matrix(c(0.5,0.4,0.1,0.2,0.1,0.7), nrow=2, ncol=3, byrow=T)
  
  lambda2_source <- initHMM(states, obs, P2_source, A2_source, B2_source)
  
  return(list(m1 = lambda1_source, m2 = lambda2_source))
}

HMM_simu <- function(hmm1, hmm2, T = 10){
  return(
    list(
      obs1 = simHMM(hmm1, T)$observation, 
      obs2 = simHMM(hmm2, T)$observation,
      states1 = simHMM(hmm1, T)$states, 
      states2 = simHMM(hmm2, T)$states
    )
  )
}

logLikelihood <- function(hmm, obs){
  fwd <- forward(hmm, obs)
  
  loglike = fwd[1, length(obs)]
  
  for(i in 2:length(hmm$States)){
    t = fwd[i, length(obs)]
    
    if (t > - Inf){
      loglike = t + log(1+exp(loglike-t))
    }
  }
  
  return(loglike)
}

HMM_new <- function(){
  states <- c("s1","s2")
  obs <- c("a","b","c")
  
  P1 <-c(1,0)
  A1 <- matrix(c(0.8,0.2,0.5,0.5), nrow=2, ncol=2, byrow=T)
  B1 <- matrix(c(0.3,0.4,0.3,0.1,0.8,0.1), nrow=2, ncol=3, byrow=T)
  
  M1 <- initHMM(states, obs, P1, A1, B1)
  M2 <- M1
  
  return(list(m1 = M1, m2 = M2))
}

checkLikelihoodBW <- function(hmm, obs){
  hmm_new <- list(hmm=hmm, difference=0)
  
  for(i in 1:10){
    hmm_new <- baumWelch(hmm_new$hmm, obs, 1)
    cat("Likelihood after iteration", i, ": ", logLikelihood(hmm_new$hmm, obs), "\n")
  }
  
  return(hmm_new)
}

train_BW <- function(training, N = 10) {
  hmms <- HMM_new()
  
  hmm1 <- baumWelch(hmms$m1,
                    training$obs1,
                    N)

  hmm2 <- baumWelch(hmms$m2,
                    training$obs2,
                    N)
  
  return (list(m1 = hmm1$hmm,
               m2 = hmm2$hmm))
}

classify <- function(source1_2, model1_2, T = 10, N = 1) {
  true_pos = false_pos = true_neg = false_neg = 0
  
  for (i in 1:(N / 2)) {
    test1_2 <- HMM_simu(source1_2$m1, source1_2$m2, T)
    
    r <- list(
      o1m1 = logLikelihood(model1_2$m1, test1_2$obs1),
      o1m2 = logLikelihood(model1_2$m2, test1_2$obs1),
      o2m1 = logLikelihood(model1_2$m1, test1_2$obs2),
      o2m2 = logLikelihood(model1_2$m2, test1_2$obs2)
    )
    
    if (r$o1m1 > r$o1m2)
      true_pos = true_pos + 1
    else
      false_neg = false_neg + 1
    if (r$o2m2 > r$o2m1)
      true_neg = true_neg + 1
    else
      false_pos = false_pos + 1
  }
  
  print(
    matrix(
      c(true_pos, false_neg, false_pos, true_neg),
      ncol = 2,
      nrow = 2,
      byrow = TRUE,
      dimnames = list(
        c("is true", "is false"), 
        c("found true", "found false")
      )
    )
  )
  
  return (true_pos + true_neg) / N
}

Plot_Reco <- function(source1_2, model1_2, Tmax = 100, N = 1) {
  reco_rate <- NULL
  samples_size <- 1:10 * (Tmax/ 10)
  
  for (i in 1:10 * (Tmax/ 10)) {
    reco_rate <- c(reco_rate, (classify(source1_2,
                               model1_2,
                               i,
                               N)) / N)
  }
  
  plot(samples_size, reco_rate, type = "l", col = "blue", xlab = "Samples size", ylab = "Recognition rate")
}

Compare_States <- function(hmm, obs, states) {
  return(
    1 - length(
      Filter(identity, states != viterbi(hmm, obs))
    )
    / length(states)
  )
}

################ PROGRAM ################

source1_2 <- HMMs_source()
train1_2 <- HMM_simu(source1_2$m1, source1_2$m2, 100)
model1_2 <- train_BW(train1_2, 100)

# cat("P(O1|lambda1) = ",
#     loglikelihood(source1_2$m1,
#                   train1_2$obs1),
#     "\n")
# cat("P(O1|lambda2) = ",
#     loglikelihood(source1_2$m2,
#                   train1_2$obs1),
#     "\n")
# cat("P(O2|lambda1)= ",
#     loglikelihood(source1_2$m1,
#                   train1_2$obs2),
#     "\n")
# cat("P(O2|lambda2) = ",
#     loglikelihood(source1_2$m2,
#                   train1_2$obs2),
#     "\n")

# checkLikelihoodBW(model1_2$m1, train1_2$obs1)
# 
# print(source1_2$m1)
# print(model1_2$m2)

# print(classify(source1_2, model1_2, 100, 1000))
# 
# Plot_Reco(source1_2, model1_2, 100, 1000)

cat("Correct proportion for source1_2 1 on O1 compared to Viterbi: ",
    Compare_States(source1_2$m1, train1_2$obs1, train1_2$states1),
    "\n")
cat("Correct proportion for model1_2 1 on O1 compared to Viterbi: ",
    Compare_States(model1_2$m1, train1_2$obs1, train1_2$states1),
    "\n")
cat("Correct proportion for source1_2 2 on O2 compared to Viterbi: ",
    Compare_States(source1_2$m2, train1_2$obs2, train1_2$states2),
    "\n")
cat("Correct proportion for model1_2 2 on O2 compared to Viterbi: ",
    Compare_States(model1_2$m2, train1_2$obs2, train1_2$states2),
    "\n")

print("Done")