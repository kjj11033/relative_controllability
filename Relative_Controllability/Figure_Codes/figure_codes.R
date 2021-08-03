
library("readr")
library("ggplot2")
library(devtools)
library('car')
library('plyr') 
library('ggpubr')
#######  Fig 2B
data_reward_bias <- read_csv("data_reward_bias.csv")
ggplot()
ggplot(data_reward_bias, aes(x=Rewloss, y=Proportion_selecting_Me)) + geom_point(color='gray',alpha = 0.5) + 
  geom_boxplot(alpha=0.3,color='black',outlier.colour=NA) + ylim(0, 1) + theme_bw() +
  scale_y_continuous(breaks=c(0,0.5,1))

#######  Fig 2C
data_opt = read_csv("data_opt.csv")

ggplot()
ggplot(data_opt, aes(x=Self_Other, y=Optimal_choice_proportion)) + geom_point(color='gray',alpha = 0.5) + 
  geom_boxplot(alpha=0.3,color='black',outlier.colour=NA) + ylim(0, 1) + theme_bw() +
  scale_y_continuous(breaks=c(0,0.5,1))

####### Fig 3D
data_bias_par = read_csv("data_bias_par.csv")

ggplot()
ggplot(data_bias_par, aes(x=index, y=parameter)) + geom_point(color='gray',alpha = 0.5) + 
  geom_boxplot(alpha=0.3,color='black',fill=c('white','white'),outlier.colour=NA) + theme_bw()

####### Fig 3E
data_tau_par = read_csv("data_tau_par.csv")

ggplot()
ggplot(data_tau_par, aes(x=index, y=tau)) + geom_point(color='gray',alpha = 0.5) + 
  geom_boxplot(alpha=0.3,color='black',fill=c('white'),outlier.colour=NA) + theme_bw()


####### Fig 4
data_illusion = read_csv("data_illusion.csv")

ggplot()
ggplot(data_illusion, aes(x=bias_pos, y=illusion_of_control)) + geom_point(color ='blue',alpha=0.3) + 
  stat_smooth(method = lm, se=TRUE, color ='black',size=0.5) + theme_bw()

data_adapt = read_csv("data_adapt.csv")

ggplot()
ggplot(data_adapt, aes(x=bias_pos, y=optimal_choice_proportion)) + geom_point(color ='blue',alpha=0.3) + 
  stat_smooth(method = lm, se=TRUE, color ='black',size=0.5) + theme_bw()


####### Fig 5A
data_striatum_activity = read_csv("data_striatum_activity.csv")
data_tpj_activity = read_csv("data_tpj_activity.csv")

df2=data_striatum_activity

striatum_activity_fig = ggplot()+
  geom_bar(data=df2,mapping=aes(x=time, y=mean),stat='identity',alpha=0.5,fill='gray') + 
  geom_errorbar(df2,mapping=aes(x=time,ymin=mean-se,ymax=mean+se),width=0.2) + theme_bw() +
  geom_hline(yintercept=0) + ylim(0,1.5)

df2=data_tpj_activity
tpj_activity_fig= ggplot()+
  geom_bar(data=df2,mapping=aes(x=time, y=mean),stat='identity',alpha=0.5,fill='gray') + 
  geom_errorbar(df2,mapping=aes(x=time,ymin=mean-se,ymax=mean+se),width=0.2) + theme_bw() +
  geom_hline(yintercept=0) + ylim(0,1.1)

###### Fig 6A
data_vmPFC_pc = read_csv("data_vmPFC_pc.csv")

df2=data_vmPFC_pc
ggplot()+
  geom_bar(data=df2,mapping=aes(x=Predicted_Controllability, y=VD_effect_in_vmPFC),stat='identity',alpha=0.5,fill='gray') + 
  geom_errorbar(df2,mapping=aes(x=Predicted_Controllability,ymin=VD_effect_in_vmPFC-se,ymax=VD_effect_in_vmPFC+se),width=0.2) + theme_bw() +
  geom_hline(yintercept=0) + ylim(0,1.5)

###### Fig 6B
data_fmr_vmpfc_tau = read_csv("data_fmr_vmpfc_tau.csv")

ggplot()
ggplot(data_fmr_vmpfc_tau, aes(x=vmPFC_VD_signal_amplification_by_controllability, y=tau)) + geom_point(color ='blue',alpha=0.3) + 
  stat_smooth(method = lm, se=TRUE, color ='black',size=0.5) + theme_bw()
