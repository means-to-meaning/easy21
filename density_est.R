
n <- 1000
x1 <- rnorm(n,mean=1)
x2 <- rnorm(n,mean=1.1)

mc_kernel_mean <- function(x, n_samples=100000, conf=FALSE) {
  if (conf) {
  iter <- 100
  bw <- density(x)$bw
  # Draw from the sample and then from the kernel
  means <- sample(x, n_samples*iter, replace = TRUE)
  mus_matrix <- matrix(rnorm(n_samples*iter, mean = means, sd = bw),ncol=iter)
  mus <- colMeans(mus_matrix) 
  return(quantile(mus,probs=c(0.05,0.5,0.95)))
  } else {
  bw <- density(x)$bw
  # Draw from the sample and then from the kernel
  means <- sample(x, n_samples, replace = TRUE)
  mu <- mean(rnorm(n_samples, mean = means, sd = bw), prob = TRUE)
  }
  return(mu)
}

mc_kernel_mean(x1,conf=T)
mc_kernel_mean(x2,conf=T)