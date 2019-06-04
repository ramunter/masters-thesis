library(ggplot2)
library(dplyr)
library(reshape2)
setwd("~/masters-thesis/data/Acrobot")
theme_set(theme_minimal() + 
                theme(legend.position = c(0, 1), 
                legend.justification = c(0, 1), 
                legend.text = element_text(size=10),
                axis.text.x = element_text(color='black'),
                axis.text.y = element_text(color='black'),
                text = element_text(size=14),
                panel.grid.major = element_line(colour = "#AAAAAA"),
                panel.grid.minor = element_line(colour = "#FFFFFF")))

plot_data = function(df){
   df2 = df %>% group_by(Step) %>% 
           summarize(mean=mean(value),
                     q1 = quantile(value, probs=0.1),
                     q9 = quantile(value, probs= 0.9))
   
   ggplot(data=df2, aes(x=Step)) + 
       geom_line(aes(y=q1), lineend= "round", color='#CCCCCC', size=1, alpha=0.7) +
       geom_line(aes(y=q9), lineend= "round", color='#CCCCCC', size=1, alpha=0.7) +       
       geom_ribbon(aes(ymin=q1, ymax=q9), fill='#CCCCCC', alpha=0.7) +
       geom_line(aes(y=mean), lineend = "round", size=1, alpha=1) + 
       geom_point(aes(y=mean), size=2) + 
       labs(x="Training Steps (Thosands)", y="Average Evaluation Reward", color = "")
}

load_data = function(method){
    df = data.frame()
    for(i in 1:10){
        filename = paste("run-", method,"_acrobot_", i-1, "-tag-Eval_AverageReturns.csv", sep='')
        print(filename)
        temp = read.csv(filename)
        temp$it = i
        df = rbind(df, temp)
    }
    df$Wall.time = NULL
    df = melt(df, measure.vars=c("Value"))
    return(df)
}

bdqn = load_data("bdqn")
dqn = load_data("dqn")

plot_data(bdqn)
ggsave("../../Thesis/fig/BDQNAcrobot.png", width=126*1, height=63*2, units="mm", dpi=150)
plot_data(dqn)
ggsave("../../Thesis/fig/DQNAcrobot.png", width=126*1, height=63*2, units="mm", dpi=150)