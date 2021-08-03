function [dat1, dat2, traj, infStates] = tapas_controllability_null_simulation(r)

% ============================================================
p = r.p_prc.p;
rp_design = r.rp_design;
RL_design = r.RL_design;
rp_ind = r.rp_ind;
setnum = r.setnum;
be_other = 1.5;
Rand_Prob = 0.8;
alpha_other_actual = 0.3;
n = length(rp_design)+1;
% Task-Generated inputs
feedback = [0; NaN(n,1)]; %[0; r.u(:,1)]; % feedback generated from the task
u_other = [0; NaN(n,1)]; %[0; r.u(:,2)]; % participants' choice, right = 1, left = 0
ctr_ind = [0; NaN(n,1)]; %[0; r.u(:,3)]; %controllability question index
ctr_ind(rp_ind) = 1;

% Real response data of participants
u_self = [0; NaN(n,1)]; %[0; r.y(:,1)]; % participants' choice, right = 1, left = 0
ctr_choice = [0; NaN(n,1)]; %

mu_computer = NaN(n,1);
muhat_computer = NaN(n,1);
rew_PE_computer = NaN(n,1);

% ============================================================

% Number of levels
l = 2; % self and other

% Unpack parameters
mu_0 = p(1:l);
% alpha_self = tapas_sgm(p(l+1),1);
% alpha_other = tapas_sgm(p(2*l),1);
alpha_self =p(l+1);
alpha_other = 0;%p(2*l);

theta= 0;%p(2*l+1);


% Initialize updated quantities

% Representations
mu = NaN(n,l);

% Other quantities
muhat = NaN(n,l);

rew_PE = NaN(n,1);

% Representation priors
% Note: first entries of the other quantities remain
% NaN because they are undefined and are thrown away
% at the end; their presence simply leads to consistent
% trial indices.
mu(1,1) = tapas_sgm(mu_0(1), 1);
mu(1,2:end) = mu_0(2:end);

% Pass through representation update loop
for k = 2:1:n
    if not(ismember(k-1, r.ign))
        
        muhat(k,:) = mu(k-1,:);
        % ============================================================
        u_self(k) = r.rew_fun(r,muhat(k,1), 0);                      
        muhat_computer(k) = mu_computer(k-1);
        u_other(k) = sigm(muhat_computer(k)-1/2,be_other);  
        feedback(k) = fbk(rp_design,RL_design,u_self,u_other,Rand_Prob,k);       
        % Update belief of participant
        if feedback(k) == 1            
           rew_PE(k,1) = u_self(k)-muhat(k,1);
           rew_PE(k,2) = u_other(k)-muhat(k,2);
           rew_PE_computer(k) = u_other(k)-muhat_computer(k);
        elseif feedback(k) == -1
           rew_PE(k,1) = (1-u_self(k))-muhat(k,1);
           rew_PE(k,2) = (1-u_other(k))-muhat(k,2);
           rew_PE_computer(k) = (1-u_other(k))-muhat_computer(k);
        end
        mu_computer(k) = muhat_computer(k,:) + alpha_other_actual.*rew_PE_computer(k,:);                
        % ============================================================
        mu(k,:) = muhat(k,:) + [alpha_self alpha_other].*rew_PE(k,:);
        
    else
        mu(k,:) = mu(k-1,:);         
        muhat(k,:) = muhat(k-1,:);
        rew_PE(k,:) = 0;        
    end
    % ============================================================
    ctr_choice(k) = r.ctr_fun(r,muhat(k,1), 0);
    % ============================================================    
end

% Check validity of trajectories
if any(isnan(mu(:)))
    error('tapas:hgf:VarApproxInvalid', 'Variational approximation invalid. Parameters are in a region where model assumptions are violated.');
else
    % Check for implausible jumps in trajectories
    dmu = diff(mu(:,2:end));
    rmdmu = repmat(sqrt(mean(dmu.^2)),length(dmu),1);
    jumpTol = 16;
    if any(abs(dmu(:)) > jumpTol*rmdmu(:))
        error('tapas:hgf:VarApproxInvalid', 'Variational approximation invalid. Parameters are in a region where model assumptions are violated.');
    end
end

% Remove other dummy initial values
muhat(1,:) = [];
mu(1,:) = [];
rew_PE(1,:) = [];

% Create result data structure
traj = struct;

traj.mu     = mu;
traj.sa     = mu.*(1-mu);

traj.muhat  = muhat;
traj.sahat  = muhat.*(1-muhat);

% Updates with respect to prediction
traj.rew_PE = rew_PE;

% Create matrices for use by the observation model
infStates = NaN(n-1,l,8);
infStates(:,:,1) = traj.muhat;
infStates(:,:,2) = traj.sahat;
infStates(:,:,3) = traj.mu;
infStates(:,:,4) = traj.sa;


% ============================================================
% Create result data structure
u_self(1,:)=[];
u_other(1,:)=[];
feedback(1,:)=[];



dat1 = struct;
dat1.response = [u_self(1:100,:) NaN(100,1)];
dat1.input = [feedback(1:100,:) u_other(1:100,:) NaN(100,1)];
dat1.rp_design = rp_design(1:100,:);
dat1.setnum = setnum;

dat2 = struct;
dat2.response = [u_self(101:200,:) NaN(100,1)];
dat2.response(rp_ind,2)=ctr_choice(rp_ind+100,1);
dat2.input = [feedback(101:200,:) u_other(101:200,:) zeros(100,1)];
dat2.input(rp_ind,3)=1;
dat2.rp_design = rp_design(101:200,:);
dat2.setnum = setnum;



function feedback = fbk(rp_design,RL_design,u_self,u_other,Rand_Prob,k)

    tmp = rand;

    if rp_design(k-1) == 1
        if (RL_design(k-1) == u_self(k)) && (tmp<Rand_Prob)
            feedback = 1;
        elseif (RL_design(k-1) ~= u_self(k)) && (tmp>Rand_Prob)
            feedback = 1;
        else
            feedback = -1;
        end
    else
        if (RL_design(k-1) == u_other(k)) && (tmp<Rand_Prob)
            feedback = 1;
        elseif (RL_design(k-1) ~= u_other(k)) && (tmp>Rand_Prob)
            feedback = 1;
        else
            feedback = -1;
        end
    end      
end

function decision = sigm(x,beta)
    decision = rand<(1./(1+exp(-beta*x)));
end
% ============================================================
end