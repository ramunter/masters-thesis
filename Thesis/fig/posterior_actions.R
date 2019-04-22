library(ggplot2)
ggplot(data=data.frame(x=c(-5, 10)), aes(x)) +
    stat_function(fun = dnorm, args=list(mean=3, sd=1.5), aes(colour="A"), size=1.5, n=100) + 
    stat_function(fun = dnorm, args=list(mean=1.5, sd=2), aes(colour="B"), size=1.5, n=100) +
    scale_colour_manual("Action", values = c("red", "blue")) + 
    xlab("Q-value") + ylab("Probability") + theme_bw()
       
