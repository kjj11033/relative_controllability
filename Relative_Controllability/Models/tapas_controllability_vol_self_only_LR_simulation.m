function [dat1, dat2, traj, infStates] = tapas_controllability_vol_self_only_LR_simulation(r)
% ============================================================
p=r.p_prc.p;
rp_design = r.rp_design;
RL_design = r.RL_design;
rp_ind = r.rp_ind;
setnum = r.setnum;
be_other = 1.5;
Rand_Prob = 0.8;
alpha_other_actual = 0.3;
n = length(rp_design);
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
l=3;
t = ones(n+1,1);
% ============================================================
% Unpack parameters
mu_0 = p(1:l);
sa_0 = p(l+1:2*l);
rho  = p(2*l+1:3*l);
ka   = p(3*l+1:4*l-1); 
om   = p(4*l:5*l-2);
th   = exp(p(5*l-1));
%% CTR parameter
theta_ctr = p(5*l);


% Representations
mu = NaN(n,l);
pi = NaN(n,l);

% Other quantities
muhat = NaN(n,l);
pihat = NaN(n,l);
v     = NaN(n,l);
w     = NaN(n,l-1);
da    = NaN(n,l);

% Representation priors
% Note: first entries of the other quantities remain
% NaN because they are undefined and are thrown away
% at the end; their presence simply leads to consistent
% trial indices.
mu(1,1) = tapas_sgm(mu_0(1), 1);
pi(1,1) = Inf;
mu(1,2:end) = mu_0(2:end);
pi(1,2:end) = 1./sa_0(2:end);

%%
ctr_mu = NaN(n,1); % initialize subjective controllability probabiltiy matrix
ctr_muhat = NaN(n,1); % initialize subjective controllability probabiltiy matrix
ctr_mu(1) = 0.5; % initial belief of p(feedback|my choice)
%%

% Number of trials (including prior)
n = length(u_self);

% Pass through representation update loop
for k = 2:1:n
    if not(ismember(k-1, r.ign))        
        ctr_muhat(k,:) = ctr_mu(k-1,:)+theta_ctr*(1/2-ctr_mu(k-1,:)); % considers drift                        
       %%
        %%%%%%%%%%%%%%%%%%%%%%
        % Effect of input u(k)
        %%%%%%%%%%%%%%%%%%%%%%
        
        % 2nd level prediction
        muhat(k,2) = mu(k-1,2) +t(k) *rho(2);
        
        % 1st level
        % ~~~~~~~~~
        % Prediction
        muhat(k,1) = tapas_sgm(muhat(k,2), 1);
        %% ============================================================
        u_self(k) = r.rew_fun(r, muhat(k,1), ctr_muhat(k,1));                      
        muhat_computer(k) = mu_computer(k-1);
        u_other(k) = sigm(muhat_computer(k)-1/2,be_other);  
        feedback(k) = fbk(rp_design,RL_design,u_self,u_other,Rand_Prob,k);       
        % Update belief of participant
        if feedback(k) == 1            
%            rpe = u_other(k)-muhat_other(k);
           rew_PE_computer(k) = u_other(k)-muhat_computer(k);
        elseif feedback(k) == -1
%            rpe = (1-u_other(k))-muhat_other(k);
           rew_PE_computer(k) = (1-u_other(k))-muhat_computer(k);
        end
        mu_computer(k) = muhat_computer(k,:) + alpha_other_actual.*rew_PE_computer(k,:);                
        u(k) = u_self(k);        
        %% ============================================================        
        % Precision of prediction
        pihat(k,1) = 1/(muhat(k,1)*(1 -muhat(k,1)));

        % Updates
        pi(k,1) = Inf;
        mu(k,1) = u(k);

        % Prediction error
        da(k,1) = mu(k,1) -muhat(k,1);

        % 2nd level
        % ~~~~~~~~~
        % Prediction: see above
        
        % Precision of prediction
        pihat(k,2) = 1/(1/pi(k-1,2) +exp(ka(2) *mu(k-1,3) +om(2)));

        % Updates
        pi(k,2) = pihat(k,2) +1/pihat(k,1);
        mu(k,2) = muhat(k,2) +1/pi(k,2) *da(k,1).*ctr_muhat(k,1); % CTR-multiplied

        % Volatility prediction error
        da(k,2) = (1/pi(k,2) +(mu(k,2) -muhat(k,2))^2) *pihat(k,2) -1;

        if l > 3
            % Pass through higher levels
            % ~~~~~~~~~~~~~~~~~~~~~~~~~~
            for j = 3:l-1
                % Prediction
                muhat(k,j) = mu(k-1,j) +t(k) *rho(j);
                
                % Precision of prediction
                pihat(k,j) = 1/(1/pi(k-1,j) +t(k) *exp(ka(j) *mu(k-1,j+1) +om(j)));

                % Weighting factor
                v(k,j-1) = t(k) *exp(ka(j-1) *mu(k-1,j) +om(j-1));
                w(k,j-1) = v(k,j-1) *pihat(k,j-1);

                % Updates
                pi(k,j) = pihat(k,j) +1/2 *ka(j-1)^2 *w(k,j-1) *(w(k,j-1) +(2 *w(k,j-1) -1) *da(k,j-1));

                if pi(k,j) <= 0
                    error('tapas:hgf:NegPostPrec', 'Negative posterior precision. Parameters are in a region where model assumptions are violated.');
                end

                mu(k,j) = muhat(k,j) +1/2 *1/pi(k,j) *ka(j-1) *w(k,j-1) *da(k,j-1);
    
                % Volatility prediction error
                da(k,j) = (1/pi(k,j) +(mu(k,j) -muhat(k,j))^2) *pihat(k,j) -1;
            end
        end

        % Last level
        % ~~~~~~~~~~
        % Prediction
        muhat(k,l) = mu(k-1,l) +t(k) *rho(l);
        
        % Precision of prediction
        pihat(k,l) = 1/(1/pi(k-1,l) +t(k) *th);

        % Weighting factor
        v(k,l)   = t(k) *th;
        v(k,l-1) = t(k) *exp(ka(l-1) *mu(k-1,l) +om(l-1));
        w(k,l-1) = v(k,l-1) *pihat(k,l-1);
        
        % Updates
        pi(k,l) = pihat(k,l) +1/2 *ka(l-1)^2 *w(k,l-1) *(w(k,l-1) +(2 *w(k,l-1) -1) *da(k,l-1));
 
        if pi(k,l) <= 0
            error('tapas:hgf:NegPostPrec', 'Negative posterior precision. Parameters are in a region where model assumptions are violated.');
        end

        mu(k,l) = muhat(k,l) +1/2 *1/pi(k,l) *ka(l-1) *w(k,l-1) *da(k,l-1).*ctr_muhat(k,1); % CTR-multiplied
    
        % Volatility prediction error
        da(k,l) = (1/pi(k,l) +(mu(k,l) -muhat(k,l))^2) *pihat(k,l) -1;
       %% ==========  Update controllability based on posterior========
        ctr_mu(k) = abs(tapas_sgm(mu(k,2),1)-1/2)*2;
        
    else
        
        ctr_mu(k) = ctr_mu(k-1);
        ctr_muhat(k) = ctr_muhat(k-1);        
       %%
        mu(k,:) = mu(k-1,:); 
        pi(k,:) = pi(k-1,:);

        muhat(k,:) = muhat(k-1,:);
        pihat(k,:) = pihat(k-1,:);
        
        v(k,:)  = v(k-1,:);
        w(k,:)  = w(k-1,:);
        da(k,:) = da(k-1,:);
       %%
        
    end
    % ============================================================
    ctr_choice(k) = r.ctr_fun(r,muhat(k,1), ctr_mu(k,1));
    % ============================================================     
end

% Implied learning rate at the first level
sgmmu2 = tapas_sgm(mu(:,2), 1);
lr1    = diff(sgmmu2)./da(2:n,1);
lr1(da(2:n,1)==0) = 0;

% Remove representation priors
mu(1,:)  = [];
pi(1,:)  = [];

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
%%
% Remove other dummy initial values
muhat(1,:) = [];
pihat(1,:) = [];
v(1,:)     = [];
w(1,:)     = [];
da(1,:)    = [];

ctr_mu(1,:) = [];
ctr_muhat(1,:) = [];

% Create result data structure
traj = struct;

traj.mu     = mu;
traj.sa     = 1./pi;

traj.muhat  = muhat;
traj.sahat  = 1./pihat;

traj.v      = v;
traj.w      = w;
traj.da     = da;

traj.ctr_mu = ctr_mu;
traj.ctr_sa = ctr_mu.*(1-ctr_mu);

traj.ctr_muhat = ctr_muhat;
traj.ctr_sahat = ctr_muhat.*(1-ctr_muhat);

% Updates with respect to prediction
traj.ud = muhat -mu;

% Psi (precision weights on prediction errors)
psi        = NaN(n-1,l);
psi(:,2)   = 1./pi(:,2);
psi(:,3:l) = pihat(:,2:l-1)./pi(:,3:l);
traj.psi   = psi;

% Epsilons (precision-weighted prediction errors)
epsi        = NaN(n-1,l);
epsi(:,2:l) = psi(:,2:l) .*da(:,1:l-1);
traj.epsi   = epsi;

% Full learning rate (full weights on prediction errors)
wt        = NaN(n-1,l);
wt(:,1)   = lr1;
wt(:,2)   = psi(:,2);
wt(:,3:l) = 1/2 *(v(:,2:l-1) *diag(ka(2:l-1))) .*psi(:,3:l);
traj.wt   = wt;

% Create matrices for use by the observation model
infStates = NaN(n-1,l,4);
infStates(:,:,1) = traj.muhat;
infStates(:,:,2) = traj.sahat;
infStates(:,:,3) = traj.mu;
infStates(:,:,4) = traj.sa;
infStates(:,1:2,5) = [traj.ctr_muhat 1-traj.ctr_muhat];
infStates(:,1:2,6) = [traj.ctr_sahat traj.ctr_sahat];
infStates(:,1:2,7) = [traj.ctr_mu 1-traj.ctr_mu];
infStates(:,1:2,8) = [traj.ctr_sa traj.ctr_sa];

%%

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
