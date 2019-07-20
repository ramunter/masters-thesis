library(ggplot2)
library(dplyr)
library(reshape2)
library(ggpubr)

theme_set(theme_minimal() + 
              theme(legend.position = 'bottom', 
                    legend.text = element_text(size=10),
                    axis.text.x = element_text(color='black'),
                    axis.text.y = element_text(color='black'),
                    text = element_text(size=14),
                    panel.grid.major = element_line(colour = "#AAAAAA"),
                    panel.grid.minor = element_line(colour = "#FFFFFF")))

# Original

setwd("~/masters-thesis/data/Stability")

df1 = read.csv("b_stability_acrobot.csv")
ggplot(data=df1, aes(x=Step)) + 
    geom_line(aes(y=Value),  size=1.5, alpha=0.8) + 
    geom_point(aes(y=Value),  size=1.5, shape=21, color="black", stroke=1, fill="white") + 
    labs(x="Training Steps", y="InvGamma b Parameter")+
    ylim(c(0,1.75e6))
ggsave("../../Thesis/fig/BUnstable.png", width=126*1, height=63*2, units="mm", dpi=150)

df2 = read.csv("b_stability_acrobot2.csv")
ggplot(data=df2, aes(x=Step)) + 
    geom_line(aes(y=Value),  size=1.5, alpha=0.8) + 
    geom_point(aes(y=Value),  size=1.5, shape=21, color="black", stroke=1, fill="white") + 
    labs(x="Training Steps", y="")+theme(axis.text.y=element_blank())+
    ylim(c(0,1.75e6))
ggsave("../../Thesis/fig/Bstable.png", width=126*1, height=63*2, units="mm", dpi=150)

df3 = read.csv("b_stability_acrobot3.csv")
ggplot(data=df3, aes(x=Step)) + 
    geom_line(aes(y=Value),  size=1.5, alpha=0.8) + 
    geom_point(aes(y=Value),  size=1.5, shape=21, color="black", stroke=1, fill="white") + 
    labs(x="Training Steps", y="")+theme(axis.text.y=element_blank())+
    ylim(c(0,1.75e6))
ggsave("../../Thesis/fig/Bstable2.png", width=126*1, height=63*2, units="mm", dpi=150)
