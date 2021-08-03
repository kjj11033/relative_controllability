%% Fig 1A - Controllability Index analysis
controllability_model = fitglme(ci_table,'causality_choice~CIx_self*CIx_other+gender+(1|subj)','distribution','binomial')

%% Load behavior data and model list
load('Data_Relative_Controllability_v2.mat');
addpath('Models');
%% Fit all Computational models
N_subj = 1;
N_model = 31;
model_estim=cell(N_model,N_subj);
for k=1:N_subj
    tresp=[dat_all{k,1}.response;dat_all{k,2}.response];
    tinp=[dat_all{k,1}.input;dat_all{k,2}.input];tinp(1:100,3)=0;
    for model_num = 1:N_model
        model_estim{k,model_num} = tapas_fitModel_controllability(tresp,tinp,model_list{model_num,1},model_list{model_num,2});
    end    
end
%% Log model evidence matrix for Bayesian model comparison
LME_all=zeros(N_subj,N_model);
for i=1:N_subj
    for j=1:N_model
        LME_all(i,j)=model_estim {i,j}.optim.LME;
    end
end
%% Parameters of individual
par_all_temp =zeros(103,10); 
for k=1:N_subj
    par_all_temp(k,:) = [model_estim{k,12}.p_prc.p model_estim{k,12}.p_obs.p];
end
% alpha_self/alpha_other/theta/bias_pos/bias_neg/beta/beta_controllability/tau
par_all=[par_all_temp(:,3:5) log(par_all_temp(:,6:7)) par_all_temp(:,8:10)];