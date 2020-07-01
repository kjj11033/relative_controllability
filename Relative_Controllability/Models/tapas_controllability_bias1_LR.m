function [traj, infStates] = tapas_controllability_bias1_LR(r, p, varargin)
% Calculates the trajectories of the agent's representations under the HGF
%
% This function can be called in two ways:
% 
% (1) tapas_hgf_binary(r, p)
%   
%     where r is the structure generated by tapas_fitModel and p is the parameter vector in native space;
%
% (2) tapas_hgf_binary(r, ptrans, 'trans')
% 
%     where r is the structure generated by tapas_fitModel, ptrans is the parameter vector in
%     transformed space, and 'trans' is a flag indicating this.
%
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2012-2013 Christoph Mathys, TNU, UZH & ETHZ
%
% This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.


% Transform paramaters back to their native space if needed
if ~isempty(varargin) && strcmp(varargin{1},'trans');
    p = tapas_controllability_bias1_LR_transp(r, p);
end

% Number of levels
l = 2; % self and other

% Unpack parameters
mu_0 = p(1:l);
% alpha_self = tapas_sgm(p(l+1),1);
% alpha_other = tapas_sgm(p(2*l),1);
alpha_self =p(l+1);
alpha_other = p(2*l);

theta= p(2*l+1);

self_bias = p(2*l+2);
% fprintf('self bias is %d \n\n',self_bias);
% Add dummy "zeroth" trial
% Task-Generated inputs
feedback = [0; r.u(:,1)]; % feedback generated from the task
u_other = [0; r.u(:,2)]; % participants' choice, right = 1, left = 0
ctr_ind = [0; r.u(:,3)]; %controllability question index

% Real response data of participants
u_self = [0; r.y(:,1)]; % participants' choice, right = 1, left = 0



% Number of trials (including prior)
n = length(u_self);

% Initialize updated quantities

% Representations
mu = NaN(n,l);
ctr_mu = NaN(n,1); % initialize subjective controllability probabiltiy matrix

% Other quantities
muhat = NaN(n,l);
ctr_muhat = NaN(n,1); % initialize subjective controllability probabiltiy matrix

rew_PE = NaN(n,1);

% Representation priors
% Note: first entries of the other quantities remain
% NaN because they are undefined and are thrown away
% at the end; their presence simply leads to consistent
% trial indices.
mu(1,1) = tapas_sgm(mu_0(1), 1);
mu(1,2:end) = mu_0(2:end);
ctr_mu(1) = 0.5; % initial belief of p(feedback|my choice)

% Pass through representation update loop
for k = 2:1:n
    if not(ismember(k-1, r.ign))
        ctr_muhat(k,:) = ctr_mu(k-1,:)+theta*(1/2-ctr_mu(k-1,:)); % considers drift         
        muhat(k,:) = mu(k-1,:);
       %% ==========  Update controllability based on posterior=========
        % Update belief of participant
        if feedback(k) == 1            
           rew_PE(k,1) = u_self(k)-muhat(k,1);
           rew_PE(k,2) = u_other(k)-muhat(k,2);
        elseif feedback(k) == -1
           rew_PE(k,1) = (1-u_self(k))-muhat(k,1);
           rew_PE(k,2) = (1-u_other(k))-muhat(k,2);
        end
        mu(k,:) = muhat(k,:) + [alpha_self*ctr_muhat(k,1) alpha_other*(1-ctr_muhat(k,1))].*rew_PE(k,:);
        z=(feedback(k)==1);
        lik_mine=(abs(u_self(k)-(1-mu(k,1)))^z)*(abs(u_self(k)-mu(k,1))^(1-z));
        lik_other=(abs(u_other(k)-(1-mu(k,2)))^z)*(abs(u_other(k)-(mu(k,2)))^(1-z));
        % calculate posterior controllability
        ctr_mu(k) = (self_bias*lik_mine*ctr_muhat(k))/(self_bias*lik_mine*ctr_muhat(k)+lik_other*(1-ctr_muhat(k)));
    else
        mu(k,:) = mu(k-1,:);         
        muhat(k,:) = muhat(k-1,:);
        ctr_mu(k) = ctr_mu(k-1);
        ctr_muhat(k) = ctr_muhat(k-1);
        rew_PE(k,:) = 0;        
    end
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
ctr_mu(1,:) = [];
ctr_muhat(1,:) = [];


% Create result data structure
traj = struct;

traj.mu     = mu;
traj.sa     = mu.*(1-mu);

traj.muhat  = muhat;
traj.sahat  = muhat.*(1-muhat);

traj.ctr_mu = ctr_mu;
traj.ctr_sa = ctr_mu.*(1-ctr_mu);

traj.ctr_muhat = ctr_muhat;
traj.ctr_sahat = ctr_muhat.*(1-ctr_muhat);

% Updates with respect to prediction
traj.rew_PE = rew_PE;

% Create matrices for use by the observation model
infStates = NaN(n-1,l,8);
infStates(:,:,1) = traj.muhat;
infStates(:,:,2) = traj.sahat;
infStates(:,:,3) = traj.mu;
infStates(:,:,4) = traj.sa;
infStates(:,:,5) = [traj.ctr_muhat 1-traj.ctr_muhat];
infStates(:,:,6) = [traj.ctr_sahat traj.ctr_sahat];
infStates(:,:,7) = [traj.ctr_mu 1-traj.ctr_mu];
infStates(:,:,8) = [traj.ctr_sa traj.ctr_sa];

return;
