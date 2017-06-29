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


compare_t_test_5 <- function(x1, x2) {
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

compare_t_test_20 <- function(x1, x2) {
  alpha <- 0.2
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

compare_t_test_40 <- function(x1, x2) {
  alpha <- 0.4
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


generate_random_sampling_params <- function() {
  n1 <- 10
  n2 <- 10
  n_mc <- 1000000
  param <- runif(1,1,10)
  param_diff <- runif(1,0,.1)
  param1 <- param
  param2 <- param + param_diff
  params_list = list()
  params_list[[1]] <- format_sampling_params(rexp, n1, n_mc, param1)
  params_list[[2]] <- format_sampling_params(rexp, n2, n_mc, param2)
  return(params_list)
}


generate_sampling_param_grid <- function() {
  n_mc <- 1000000
  
  # generate grid parameters for exp - exp distribution pair
  dist_func1 <- "rexp"
  dist_func2 <- "rexp"
  n1_list <- 10^(1:3)
  n2_list <- 10^(1:3)
  expectation_list <- c(1, 5, 10)
  expectation_diff_list <- c(-1, -0.1, -0.01, 0.01, 0.1,  1)
  param_grid <- list()
  param_combination_count  <- 0
  for (n1 in n1_list) {
    for (n2 in n2_list) {
      for (expectation in expectation_list) {
        for (expectation_diff in expectation_diff_list) {
          param_combination_count <- param_combination_count + 1
          params_list = list()
          expectation1 <- expectation
          expectation2 <- expectation + expectation_diff
          params_list[[1]] <- format_sampling_params(dist_func1, n1, n_mc, expectation1)
          params_list[[2]] <- format_sampling_params(dist_func2, n2, n_mc, expectation2)
          param_grid[[param_combination_count]] <- params_list
        }
      }
    }
  }
  return(param_grid)
}

format_sampling_params <- function(dist_func, n, n_mc, expectation) {
  params_list <- list()
  params_list[['dist_func']] <- dist_func
  params_list[['expectation']] <- expectation
  params_list[['n']] <- n
  rate <- 1/expectation
  params_list[['sampling_params']] <- list(n=n, rate=rate)
  params_list[['mc_sampling_params']] <- list(n=n_mc, rate=rate)
  return(params_list)
}

generate_samples <- function(params_list) {
  sampling_func1 <- eval(parse(text=params_list[[1]][['dist_func']]))
  sampling_func2 <- eval(parse(text=params_list[[2]][['dist_func']]))
  sampling_params1 <- params_list[[1]][['sampling_params']]
  sampling_params2 <- params_list[[2]][['sampling_params']]
  mc_sampling_params1 <- params_list[[1]][['mc_sampling_params']]
  mc_sampling_params2 <- params_list[[2]][['mc_sampling_params']]
  x1 <- do.call(sampling_func1, sampling_params1)
  x2 <- do.call(sampling_func2, sampling_params2)
  return(list(x1=x1, x2=x2))
}


get_ground_truth <- function(params_list, mc_mean_est) {
  if (mc_mean_est) {
      sampling_func1 <- eval(parse(text=params_list[[1]][['dist_func']]))
      sampling_func2 <- eval(parse(text=params_list[[2]][['dist_func']]))
      mc_sampling_params1 <- params_list[[1]][['mc_sampling_params']]
      mc_sampling_params2 <- params_list[[2]][['mc_sampling_params']]
      mc_expectation1 <- mean(do.call(sampling_func1, mc_sampling_params1))
      mc_expectation2 <- mean(do.call(sampling_func2, mc_sampling_params2))
      if (mc_expectation1 > mc_expectation2) {
        return(1)    
      } else if (mc_expectation1 < mc_expectation2) {
        return(2)
      } else if (mc_expectation1 == mc_expectation2) {
        return(0)
      } else {
        stop("Cannot compare distribution means")
      }
  } else if (!mc_mean_est) {
    if (params_list[[1]][["expectation"]] > params_list[[2]][["expectation"]]) {
        return(1)
    } else if (params_list[[1]][["expectation"]] < params_list[[2]][["expectation"]]) {
        return(2)
    } else if (params_list[[1]][["expectation"]] == params_list[[2]][["expectation"]]) {
        return(0)
    } else {
      stop("Cannot compare distribution means")
    }
  } else {
    print("Exiting. True parameters of distribution unknown!")
  }
}

run_games <- function(params_list, compare_groups_models) {
  nrepeat <- 1000
  mc_mean_est <- FALSE
  ntests <- length(compare_groups_models)
  res <- matrix(0,nrow=nrepeat,ncol=ntests+1)
  colnames(res) <- c("ground_truth", compare_groups_models)
  gt <- get_ground_truth(params_list, mc_mean_est)
  for (run in 1:nrepeat) {
    mysample <- generate_samples(params_list)
    # decide which group has a greater expectation
    res[run, "ground_truth"] <- gt
    for (group_comp_func_str in compare_groups_models) {
      fun <- eval(parse(text=group_comp_func_str))
      res[run, group_comp_func_str] <- fun(mysample$x1, mysample$x2)
    }
  }
  return(res)
}

get_param_description <- function(params_list) {
  res <- c(params_list[[1]][['dist_func']],
           params_list[[2]][['dist_func']],
           params_list[[1]][['n']],
           params_list[[2]][['n']])
  if ("expectation" %in% names(params_list[[1]]) && "expectation" %in% names(params_list[[2]])) {
      res <- c(res, params_list[[1]][['expectation']], params_list[[2]][['expectation']])
      return(res)
  } else if ("mc_expectation" %in% names(params_list[[1]]) && "mc_expectation" %in% names(params_list[[2]])) {
      res <- c(res, params_list[[1]][['mc_expectation']], params_list[[2]][['mc_expectation']])
      return(res)
  } else {
    stop("Error. Either the theoretical expectation, or the MC expectation estimate should be present!")
  }
}

get_performance <- function(preds, gt) {
  tp <- length(which(preds_logic) %in% which(gt_logic))
  fp <- length(!(which(preds_logic) %in% which(gt_logic)))
  tn <- length(which(!preds_logic) %in% which(!gt_logic))
  fn <- length(!(which(!preds_logic) %in% which(!gt_logic)))
  precision <- tp / (tp + fp)
  recall <- tp / (tp + fn)
  f1 <- (2 * tp * fp) / (tp + fp)
  res <- list(precision=precision, recall=recall, f1=f1)
  return(res)
}

set.seed(42)
perf_measure <- "prob_success"
undecided_count <- "undecided_proportion"
compare_groups_models <- c("compare_samplemeans", "compare_samplemedians", "compare_t_test_5", "compare_t_test_20", "compare_t_test_40")
desc_params_names <- c("dist1", "dist2", "n1", "n2", "expectation1", "expectation2")
param_grid <- generate_sampling_param_grid()
grid_result <- matrix(0,ncol=length(compare_groups_models)*2+6, nrow=length(param_grid))
colnames(grid_result) <- c(desc_params_names, paste0(compare_groups_models, '_',perf_measure), paste0(compare_groups_models, '_', undecided_count))
grid_result <- as.data.frame(grid_result)
row_count <- 0
for (params_list in param_grid) {
  print(paste0(round(row_count/length(param_grid),4)*100, "%"))
  row_count <- row_count + 1
  param_desc_values <- get_param_description(params_list)
  grid_result[row_count, desc_params_names] <- param_desc_values
  res <- run_games(params_list, compare_groups_models)
  gt <- res[,"ground_truth"]
  for (group_comp_func_str in compare_groups_models) {
    preds <- res[, group_comp_func_str]
    # consider only the decisions when the model was confident
    undecided_proportion <- sum(preds == 0) / length(preds)
    gt <- gt[preds != 0]
    preds <- preds[preds != 0]
    probability_choose_bigger_group <- sum(preds==gt)/length(gt)
    grid_result[row_count, paste0(group_comp_func_str, '_', perf_measure)] <- probability_choose_bigger_group
    grid_result[row_count, paste0(group_comp_func_str, '_', undecided_count)] <- undecided_proportion
  }
}

# save(grid_result, file="result.RData")
