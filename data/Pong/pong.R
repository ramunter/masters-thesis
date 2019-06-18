library(ggplot2)
library(dplyr)
library(reshape2)
library(jsonlite)
setwd("~/masters-thesis/data/Pong")
theme_set(theme_minimal() + 
              theme(legend.position = 'bottom', 
                    legend.text = element_text(size=10),
                    axis.text.x = element_text(color='black'),
                    axis.text.y = element_text(color='black'),
                    text = element_text(size=14),
                    panel.grid.major = element_line(colour = "#AAAAAA"),
                    panel.grid.minor = element_line(colour = "#FFFFFF")))

dopamine = fromJSON("pong.json")
dopamine = dopamine[1:3980,] # For some reason last row is NA
dopamine$Seed = as.factor(rep(seq(1,20), each=199))

dqn = dopamine %>% filter(Agent=="DQN") %>% filter(Iteration < 35)

dqn_summary = dqn %>% group_by(Iteration) %>% 
    summarize(median=median(Value),
              q1 = quantile(Value, probs=0),
              q9 = quantile(Value, probs= 1))

plot_results = function(filename, save_as){
    bdqn = read.csv(filename)#'run-bdqn_pong_1-tag-Eval_AverageReturns.csv')
    bdqn$Wall.time = NULL
    names(bdqn)= c("Iteration", "Value")
    bdqn$Agent = "BDQN"
    bdqn$Seed = 0
    
    df = rbind(dqn, bdqn)
    p1 = ggplot(data=dqn_summary, aes(x=Iteration)) +
        geom_line(aes(y=q1), lineend= "round", color='#CCCCCC',size=1, alpha=0.7) +
        geom_line(aes(y=q9), lineend= "round", color='#CCCCCC',size=1, alpha=0.7) +
        geom_ribbon(aes(ymin=q1, ymax=q9), fill='#CCCCCC', alpha=0.7) +
        geom_line(aes(y=median), lineend = "round", size=1, alpha=1) +
        geom_point(aes(y=median), size=2) +
        labs(x="Training Steps (Millions)", y="Average Evaluation Reward", color = "")
    p1 + geom_line(data=bdqn, aes(x=Iteration, y=Value), size=1.5, color='orange') +
        geom_point(data=bdqn, aes(x=Iteration, y=Value), size=3)
    
    ggplot() + geom_line(data=dqn, aes(x=Iteration, y=Value, group=Seed, color="DQN"), size=1.5, alpha=0.3) +
        geom_line(data=bdqn, aes(x=Iteration, y=Value, color="BNIG"), size=1.5) +
        geom_point(data=bdqn, aes(x=Iteration, y=Value), size=2.5) + 
        scale_colour_manual(name = "", values=c(DQN="black", BNIG="orange")) +
        labs(x="Training Steps (Millions)", y="Average Evaluation Reward", color = "")
    
    ggsave(paste("../../Thesis/fig/", save_as, ".png", sep=""), width=126*1, height=126*1, units="mm", dpi=150)
}

plot_results('run-bdqn_pong_1-tag-Eval_AverageReturns.csv', 'BDQNPong')
#plot_results('run-bdqn_pong_2-tag-Eval_AverageReturns.csv', "BDQNPongHighAlpha")
