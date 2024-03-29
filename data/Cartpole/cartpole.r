library(ggplot2)
library(dplyr)
library(reshape2)
setwd("~/masters-thesis/data/Cartpole")
theme_set(theme_minimal() + 
                theme(legend.position = 'bottom', 
                legend.text = element_text(size=10),
                axis.text.x = element_text(color='black'),
                axis.text.y = element_text(color='black'),
                text = element_text(size=14),
                panel.grid.major = element_line(colour = "#AAAAAA"),
                panel.grid.minor = element_line(colour = "#FFFFFF")))

plot_summary = function(df){
   df2 = df %>% group_by(Step) %>% 
           summarize(mean=median(value),
                     q1 = quantile(value, probs=0.1),
                     q9 = quantile(value, probs= 0.9))
   
   ggplot(data=df2, aes(x=Step)) + 
       geom_line(aes(y=q1), lineend= "round", color='#CCCCCC',size=1, alpha=0.7) +
       geom_line(aes(y=q9), lineend= "round", color='#CCCCCC',size=1, alpha=0.7) +       
       geom_ribbon(aes(ymin=q1, ymax=q9), fill='#CCCCCC', alpha=0.7) +
       geom_line(aes(y=mean), lineend = "round", size=1, alpha=1) + 
       geom_point(aes(y=mean), size=2) + 
       labs(x="Training Steps (Thosands)", y="Average Evaluation Reward", color = "")
}

plot_per_run = function(df){
    ggplot(data=df, aes(x=Step)) + 
        geom_line(aes(y=value, color=seed),  size=1, alpha=1) + 
        geom_point(aes(y=value, color=seed), size=2) + 
        labs(x="Training Steps (Thosands)", y="Average Evaluation Reward") +
        facet_wrap(~seed, ncol=2)
}

load_data = function(method){
    df = data.frame()
    for(i in 1:10){
        filename = paste("run-", method,"_cartpole_", i-1, "-tag-Eval_AverageReturns.csv", sep='')
        temp = read.csv(filename)
        temp$seed = as.factor(i)
        df = rbind(df, temp)
    }
    df$Wall.time = NULL
    df = melt(df, measure.vars=c("Value"))
    return(df)
}
setwd("~/masters-thesis/data/Cartpole/og")
bdqn = load_data("bdqn")
dqn = load_data("dqn")

plot_summary(bdqn)
ggsave("../../../Thesis/fig/BDQNCartpole.png", width=126*1, height=63*2, units="mm", dpi=150)
plot_summary(dqn)
ggsave("../../../Thesis/fig/DQNCartpole.png", width=126*1, height=63*2, units="mm", dpi=150)

plot_per_run(dqn)
ggsave("../../../Thesis/fig/PerDQNCartpole.png", width=126*1, height=63*4, units="mm", dpi=150)
plot_per_run(bdqn)
ggsave("../../../Thesis/fig/PerBDQNCartpole.png", width=126*1, height=63*4, units="mm", dpi=150)

setwd("~/masters-thesis/data/Cartpole/alpha")

bdqn = load_data("cartpole_bdqn")
plot_summary(bdqn)
ggsave("../../../Thesis/fig/AlphaBDQNCartpole.png", width=126*1, height=63*2, units="mm", dpi=150)
plot_per_run(bdqn)
ggsave("../../../Thesis/fig/AlphaPerBDQNCartpole.png", width=126*1, height=63*4, units="mm", dpi=150)

setwd("~/masters-thesis/data/Cartpole/relative")

bdqn = load_data("cartpole_plot_data3_bdqn")
plot_summary(bdqn)
ggsave("../../../Thesis/fig/RelativeBDQNCartpole.png", width=126*1, height=63*2, units="mm", dpi=150)
plot_per_run(bdqn)
ggsave("../../../Thesis/fig/RelativePerBDQNCartpole.png", width=126*1, height=63*4, units="mm", dpi=150)