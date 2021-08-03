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


%% Parameter recovery simulation
clc
niter=5000;
Nmodel = size(model_list,1);
imodel = 12; % winning MABC model

ptrans_orig=zeros(niter,8);
ptrans_recov=zeros(niter,8);
p_orig=zeros(niter,8);
p_recov=zeros(niter,8);

p_trans_est_prc=[];
p_trans_est_obs=[];
for i=1:103
    p_trans_est_prc(i,:)=model_estim{i,imodel}.p_prc.ptrans; 
    p_trans_est_obs(i,:)=model_estim{i,imodel}.p_obs.ptrans;
end    
p_trans_dist_prc_mean=mean(p_trans_est_prc);
p_trans_dist_prc_std=std(p_trans_est_prc);
p_trans_dist_obs_mean=mean(p_trans_est_obs);
p_trans_dist_obs_std=std(p_trans_est_obs);
for iter = 1:niter
    ptrans_temp_prc = mvnrnd(p_trans_dist_prc_mean,p_trans_dist_prc_std);    
    ptrans_temp_obs = mvnrnd(p_trans_dist_obs_mean,p_trans_dist_obs_std);    
    ptrans_temp=[ptrans_temp_prc ptrans_temp_obs];
    tmpname_prc=strsplit(model_list{imodel,1},'_config');
    [prc_pvec, dummy] = eval([tmpname_prc{1} ,'_transp([], ptrans_temp_prc)']);
    tmpname_obs=strsplit(model_list{imodel,2},'_config');
    [obs_pvec, dummy] = eval([tmpname_obs{1} ,'_transp([], ptrans_temp_obs)']);       

    p=[prc_pvec obs_pvec];
    
    ptrans_orig(iter,:)=ptrans_temp(3:end);
    p_orig(iter,:)=p(3:end);
    
    if iter<=niter/2
        rp_design = [dat_all{1,1}.rp_design;dat_all{1,2}.rp_design];
        RL_design = [dat_all{1,1}.rp_design;RL_design_set1];
        setnum=1;
    else
        rp_design = [dat_all{2,1}.rp_design;dat_all{2,2}.rp_design];
        RL_design = [dat_all{2,1}.rp_design;RL_design_set2];
        setnum=2;
    end
    % gen response
    err_flag = 0;

    try     
        sim = tapas_simModel_controllability(tmpname_prc{1}, prc_pvec, tmpname_obs{1}, obs_pvec,rp_design, RL_design, rp_ind', setnum);    
    catch
        err_flag = 1;
        disp('error');
    end
    while err_flag == 1
        try 
            ptrans_temp_prc = mvnrnd(p_trans_dist_prc_mean,p_trans_dist_prc_std);    
            ptrans_temp_obs = mvnrnd(p_trans_dist_obs_mean,p_trans_dist_obs_std);    
            tmpname_prc=strsplit(model_list{imodel,1},'_config');
            [prc_pvec, dummy] = eval([tmpname_prc{1} ,'_transp([], ptrans_temp_prc)']);
            tmpname_obs=strsplit(model_list{imodel,2},'_config');
            [obs_pvec, dummy] = eval([tmpname_obs{1} ,'_transp([], ptrans_temp_obs)']);                
            sim = tapas_simModel_controllability(tmpname_prc{1}, prc_pvec, tmpname_obs{1}, obs_pvec,rp_design, RL_design, rp_ind', setnum);
            err_flag = 0;
        catch
            err_flag = 1;
        end
    end
    
    dat_sim1 = sim.dat1;
    dat_sim2 = sim.dat2;
    % re-fit
    tresp_pr{iter}=[dat_sim1.response;dat_sim2.response];
    tinp_pr{iter}=[dat_sim1.input;dat_sim2.input];%tinp(1:100,3)=0;               
end    
parfor iter = 1:niter
    est_temp_re=tapas_fitModel_controllability(tresp_pr{iter},tinp_pr{iter},model_list{imodel,1},model_list{imodel,2});
    ptrans_recov(iter,:) = [est_temp_re.p_prc.ptrans(3:end) est_temp_re.p_obs.ptrans];
    p_recov(iter,:) = [est_temp_re.p_prc.p(3:end) est_temp_re.p_obs.p];
end
%% Model recovery simulation
lme_model_recov_all=cell(Nmodel,1);
Nmodel = 31;
for imodel = 1:Nmodel
    lme_model_recov_xx=zeros(niter,size(model_list,2));  % 2nd dim: models fitted after simulation 3nd dim: simulated model
    p_trans_est_prc=[];
    p_trans_est_obs=[];
    for i=1:103
        p_trans_est_prc(i,:)=model_estim{i,imodel}.p_prc.ptrans; 
        p_trans_est_obs(i,:)=model_estim{i,imodel}.p_obs.ptrans;
    end    
    p_trans_dist_prc_mean=mean(p_trans_est_prc);
    p_trans_dist_prc_std=std(p_trans_est_prc);
    p_trans_dist_obs_mean=mean(p_trans_est_obs);
    p_trans_dist_obs_std=std(p_trans_est_obs);
    for iter = 1:niter
        ptrans_temp_prc = mvnrnd(p_trans_dist_prc_mean,p_trans_dist_prc_std);    
        ptrans_temp_obs = mvnrnd(p_trans_dist_obs_mean,p_trans_dist_obs_std);    
        tmpname_prc=strsplit(model_list{imodel,1},'_config');
        [prc_pvec, dummy] = eval([tmpname_prc{1} ,'_transp([], ptrans_temp_prc)']);
        tmpname_obs=strsplit(model_list{imodel,2},'_config');
        [obs_pvec, dummy] = eval([tmpname_obs{1} ,'_transp([], ptrans_temp_obs)']);       
        
        p=[prc_pvec obs_pvec];

        if iter<=niter/2
            rp_design = [dat_all{1,1}.rp_design;dat_all{1,2}.rp_design];
            RL_design = [dat_all{1,1}.rp_design;RL_design_set1];
            setnum=1;
        else
            rp_design = [dat_all{2,1}.rp_design;dat_all{2,2}.rp_design];
            RL_design = [dat_all{2,1}.rp_design;RL_design_set2];
            setnum=2;
        end
        % gen response
        err_flag = 0;

        try     
            sim = tapas_simModel_controllability(tmpname_prc{1}, prc_pvec, tmpname_obs{1}, obs_pvec,rp_design, RL_design, rp_ind', setnum);    
        catch
            err_flag = 1;
            disp('error');
        end
        while err_flag == 1
            try 
                ptrans_temp_prc = mvnrnd(p_trans_dist_prc_mean,p_trans_dist_prc_std);    
                ptrans_temp_obs = mvnrnd(p_trans_dist_obs_mean,p_trans_dist_obs_std);    
                tmpname_prc=strsplit(model_list{imodel,1},'_config');
                [prc_pvec, dummy] = eval([tmpname_prc{1} ,'_transp([], ptrans_temp_prc)']);
                tmpname_obs=strsplit(model_list{imodel,2},'_config');
                [obs_pvec, dummy] = eval([tmpname_obs{1} ,'_transp([], ptrans_temp_obs)']);                
                sim = tapas_simModel_controllability(tmpname_prc{1}, prc_pvec, tmpname_obs{1}, obs_pvec,rp_design, RL_design, rp_ind', setnum);
                err_flag = 0;
            catch
                err_flag = 1;
            end
        end
       
        dat_sim1 = sim.dat1;
        dat_sim2 = sim.dat2;
        % re-fit
        tresp=[dat_sim1.response;dat_sim2.response];
        tinp=[dat_sim1.input;dat_sim2.input];%tinp(1:100,3)=0;
        model_recov = cell(Nmodel,1);
        
        parfor model_num = 1:Nmodel
            model_recov{model_num} = tapas_fitModel_controllability(tresp,tinp,model_list{model_num,1},model_list{model_num,2});        
            lme_model_recov_xx(iter,model_num)=model_recov{model_num}.optim.LME;            
        end
        lme_model_recov_all{imodel}=lme_model_recov_xx;
    end        
end
