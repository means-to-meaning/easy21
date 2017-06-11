
# Which distribution do we prefer drawing from to get a higher value?

# Setting A: We consider the expected value - i.e. the notion of doing a single draw many times over. Risk aversion doesn't matter here!!
# Solution: We estimate the expected value of each distribution and select one with higher value. Or if uncertain we don't draw at all.
# Do we have a measure of certainty about estimate of the expected value?

# Setting B: We consider doing a single draw once - not in the infinite limit. Does the solution compared to above? 
# How do we incorporate risk? Utility function - return vs. Probability of return

# Extension:
# What is a good procedure to sample additional data before making a decision which distribution to choose?

compare_samplemeans <- function(x1, x2) {
  sample_mean1 <- mean(x1)
  sample_mean2 <- mean(x2)
  if (sample_mean1 > sample_mean2) {
    return (1)
  } else if (sample_mean1 <= sample_mean2) {
    return(2)
  } else {
    stop("Error in sample means comparison!")
  }
}

compare_samplemedians <- function(x1, x2) {
  sample_median1 <- median(x1)
  sample_median2 <- median(x2)
  if (sample_median1 > sample_median2) {
    return (1)
  } else if (sample_median1 <= sample_median2) {
    return(2)
  } else {
    stop("Error in sample median comparison!")
  }
}


compare_t_test <- function(x1, x2) {
  alpha <- 0.05
  res_t_x1greater <- t.test(x1,x2,alternative="greater", conf.level = (1 - alpha))
  res_t_x2greater <- t.test(x2,x1,alternative="greater", conf.level = (1 - alpha))
  x1_test_greater <- (res_t_x1greater$p.value < alpha)
  x2_test_greater <- (res_t_x2greater$p.value < alpha)
  if (x1_test_greater && !x2_test_greater) {
    # if x1 bigger
    return(1)
  } else if(x2_test_greater && !x1_test_greater) {
    return(2)
  } else if(x1_test_greater && x2_test_greater) {
    if (res_t_x1greater$p.value < res_t_x2greater$p.value) {
      return(0)
    } else {
      return(0)
    }
  # if both tests return inconclusive
  } else {
    return(0)
  }
}

generate_data <- function(sampling_params, mc_mean_est=TRUE) {
  param1 <- param
  param2 <- param + param_diff
  x1 <- rexp(n1, param1)
  x2 <- rexp(n2, param2)
  expectation1 <- mean(rexp(n_mc, param1))
  expectation2 <- mean(rexp(n_mc, param2))
  
}

generate_sampling_params <- function() {
  n1 <- 10
  n2 <- 10
  n_mc <- 1000000
  param <- runif(1,1,10)
  param_diff <- runif(1,0,.1)
  param1 <- param
  param2 <- param + param_diff
  params <- list()
  params[['dist_func1']] <- 'rexp'
  params[['dist_func2']] <- 'rexp'
  params[['mu1']] <- 1/param1
  params[['mu2']] <- 1/param2
  params[['sampling_params1']] <- list(n=n1, rate=param1)
  params[['sampling_params2']] <- list(n=n2, rate=param2)
  params[['sampling_params_mc1']] <- list(n=n_mc, rate=param1)
  params[['sampling_params_mc2']] <- list(n=n_mc, rate=param2)
  return(params)
}

generate_samples <- function(params,mc_mean_est) {
  sampling_func1 <- eval(parse(text=params[['dist_func1']]))
  sampling_func2 <- eval(parse(text=params[['dist_func2']]))
  x1 <- do.call(sampling_func1,params[['sampling_params1']])
  x2 <- do.call(sampling_func2,params[['sampling_params2']])
  if (mc_mean_est) {
    mu1 <- mean(do.call(sampling_func1,params[['sampling_params_mc1']]))
    mu2 <- mean(do.call(sampling_func2,params[['sampling_params_mc2']]))
    return(list(x1=x1, x2=x2, mu1=mu1, mu2=mu2))
  }
  return(list(x1=x1, x2=x2))
}


get_gt <- function(params, mysample, mc_mean_est) {
  if (mc_mean_est) {
    if ("mu1" %in% names(mysample) && "mu2" %in% names(mysample)) {
      if (mysample[["mu1"]] > mysample[["mu2"]]) {
        return(1)    
      } else if (mysample[["mu1"]] < mysample[["mu2"]]) {
        return(2)
      } else if (mysample[["mu1"]] == mysample[["mu2"]]) {
        return(0)
      } else {
        stop("Cannot compare distribution means")
      }
    } else {
      print("Exiting. MC parameters of distribution unknown!")
    } 
  } else if (!mc_mean_est) {
    if ("mu1" %in% names(params) && "mu2" %in% names(params)) {
      if (params[["mu1"]] > params[["mu2"]]) {
        return(1)    
      } else if (params[["mu1"]] < params[["mu2"]]) {
        return(2)
      } else if (params[["mu1"]] == params[["mu2"]]) {
        return(0)
      } else {
        stop("Cannot compare distribution means")
      }
    } else {
      print("Exiting. True parameters of distribution unknown!")
    }
  }
}

run_games <- function() {
  nrepeat <- 1000
  ntests <- 3
  res <- matrix(0,nrow=nrepeat,ncol=ntests+1)
  for (run in 1:nrepeat) {
    mc_mean_est <- FALSE
    params <- generate_sampling_params()
    mysample <- generate_samples(params,mc_mean_est)
  #  plot(density(mysample$x1),col=3)
  #  lines(density(mysample$x2),col=2)
  
    res[run, 1] <- get_gt(params, mysample,mc_mean_est)
    res[run, 2] <- compare_samplemeans(mysample$x1, mysample$x2)
    res[run, 3] <- compare_samplemedians(mysample$x1, mysample$x2)
    res[run, 4] <- compare_t_test(mysample$x1, mysample$x2)
  }
  return(res)
}


set.seed(42)

res <- run_games()
table(Truth = res[,1], Prediction = res[,3])