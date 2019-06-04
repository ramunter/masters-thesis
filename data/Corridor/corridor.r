library(ggplot2)
setwd("~/masters-thesis/data/Corridor")
theme_set(theme_minimal() + 
              theme(legend.position = c(0, 1), 
                    legend.justification = c(0, 1), 
                    legend.text = element_text(size=10),
                    axis.text.x = element_text(color='black'),
                    axis.text.y = element_text(color='black'),
                    text = element_text(size=14),
                    panel.grid.major = element_line(colour = "#AAAAAA"),
                    panel.grid.minor = element_line(colour = "#FFFFFF")))

plot_data = function(df, title){
   ggplot(data=df, aes(x=N, y=Episodes.to.Learn)) + 
        geom_line(lineend = "round", size=1, alpha=0.7) + 
        geom_point(size=2, alpha=0.7) + 
        labs(y="Episodes To Learn", color = "")  + 
        facet_wrap(~variable, nrow=5)
}

df1 = read.csv("all_short_1_step.csv")
df1$variable = relevel(df1$variable, "Deep BN")
df1$variable = relevel(df1$variable, "BN")
df1$variable = relevel(df1$variable, "E Greedy")
df2 = read.csv("all_short_3_step.csv")
df2$variable = relevel(df2$variable, "Deep BN")
df2$variable = relevel(df2$variable, "BN")
df2$variable = relevel(df2$variable, "E Greedy")

plot_data(df1, "Corridor 1-Step Methods")
ggsave("../../Thesis/fig/Corridor1StepMethods.png", width=126*1, height=63*5, units="mm", dpi=150)
plot_data(df2, "Corridor 3-Step Methods")
ggsave("../../Thesis/fig/Corridor3StepMethods.png", width=126*1, height=63*5, units="mm", dpi=150)