function val = restrictfun(f, arg, free_idx, free_arg)
% This is a helper function for the construction of file handles to
% restricted functions.
%
% It returns the value of a function restricted to subset of the
% arguments of the input function handle. The input handle takes
% *one* vector as its argument.
% 
% INPUT:
%   f            The input function handle
%   arg          The argument vector for the input function containing the
%                fixed values of the restricted arguments (plus dummy values
%                for the free arguments)
%   free_idx     The index numbers of the arguments that are not restricted
%   free_arg     The values of the free arguments

% Replace the dummy arguments in arg
arg(free_idx) = free_arg;

% Evaluate
val = f(arg);

return;