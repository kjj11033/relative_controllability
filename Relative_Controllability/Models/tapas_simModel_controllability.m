function r = tapas_simModel_controllability(prc_model, prc_pvec, obs_model, obs_pvec,rp_design,RL_design,rp_ind,setnum)


% Initialize data structure to be returned
% r = struct;
r = struct;
r.rp_design = rp_design;
r.RL_design = RL_design;
r.rp_ind = rp_ind;
r.setnum = setnum;
    
% Remember perceptual model
r.c_sim.prc_model = prc_model;

% Store perceptual parameters

r.p_prc = struct;
r.p_prc.p = prc_pvec;

r.p_obs   = struct;
r.p_obs.p = obs_pvec;
% Read configuration of perceptual model
try
    prc_config_fun = str2func([prc_model, '_simulation']);
    r.c_prc = prc_config_fun();
catch
    r.c_prc = [];
end

% Get function handle to perceptual model
prc_fun = str2func([prc_model, '_simulation']);

if strcmp(obs_model, 'tapas_softmax_binary_controllability_null')
    r.rew_fun = @rewfun_null;
    r.ctr_fun = @ctrfun_null;
elseif strcmp(obs_model, 'tapas_softmax_binary_controllability_2beta')
    r.rew_fun = @rewfun_null;
    r.ctr_fun = @ctrfun_2beta; 
elseif strcmp(obs_model, 'tapas_softmax_binary_controllability_2beta_tau')
    r.rew_fun = @rewfun_tau;
    r.ctr_fun = @ctrfun_2beta;       
else
    disp('error, abnormal obs func')
end
    
r.ign = 1e10;
% Compute perceptual states
[r.dat1, r.dat2, r.traj,r.infStates] = prc_fun(r);

function u_self = rewfun_null(r,muhat, ctr_muhat)
    beta = r.p_obs.p(1);
    prob = tapas_sgm(beta*(muhat-1/2),1);
    u_self = binornd(1, prob);
end

function u_self = rewfun_tau(r, muhat, ctr_muhat)
    beta = r.p_obs.p(1);
    be_mod = r.p_obs.p(3);
    beta2 = exp(log(beta)+be_mod*log(ctr_muhat./(1-ctr_muhat)));
    prob = tapas_sgm(beta2*(muhat-1/2),1);
    u_self = binornd(1, prob);        
end

function ctr_choice = ctrfun_null(r, muhat, ctr_muhat)
    ctr_choice = binornd(1, 1/2);
end

function ctr_choice = ctrfun_2beta(r, muhat, ctr_muhat)
    beta_ctr = r.p_obs.p(2);
    prob = tapas_sgm(beta_ctr*(ctr_muhat-1/2),1);
    ctr_choice = binornd(1, prob);    
end


end


