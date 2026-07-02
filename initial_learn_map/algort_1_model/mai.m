function [model,Deltay] = mai(xh,yh,G)
% xh - modulo M input
% yh - binary output
% G - input group size
%
%  Example 1: not gate
%     xh = [0; 1]; yh = [1; 0]; % dataset
%     [model,Deltay] = mai(xh,yh) % training
%     y = model.infer(xh) % inference
%
%  Example 2: and gate
%     xh = [0 0; 0 1; 1 0; 1 1]; yh = [0; 0; 0; 1]; % dataset
%     [model,Deltay] = mai(xh,yh) % training
%     y = model.infer(xh) % inference

%#ok<*AGROW>

% Cardinality.
S = numel(yh); % number of samples
M = next_prime(max(max(xh(:))+1,S)); % modulus

% Input arguments.
if nargin < 3, G = 2; end

% Initialization.
X = {xh}; % starting input set
l = 0; % layer counter
lb = log2(M-1); % maximum layer depth
Deltay = yh; % starting output error
W = []; % starting parameter set
L = zeros(1,0); % parameter layer
I = zeros(G,0); % parameter input
while ~isempty(X{l+1})
   % Input groups.
   N = size(X{l+1},2); % number of current layer inputs
   if G < N
      Gs = nchoosek(1:N,G);
   else
      Gs = 1:N;
   end
   ng = size(Gs,2); % current group size
   
   % Refined input.
   if l <= lb
      X{l+2} = rem(floor(xh/2^(lb-l-1)),2);
   else
      X{l+2} = [];
   end
   
   % Step.
   for g = 1:size(Gs,1)
      xs = X{l+1}(:,Gs(g,:))*2.^(0:size(Gs,2)-1)'; % input groups
      V = modular_residue(cumprod([ones(S,1),repmat(xs,[1,S-1])],2),M); % Vandermond matrix
      [~,~,ju] = unique(xs); Deltayp = accumarray(ju,Deltay,[S,1],@(x)min(x)); Deltayp = Deltayp(ju); % consistent error
      if any(Deltayp)
         w = V\Deltayp; % increamental training
         Deltay = Deltay - Deltayp; % output error reduction
         W = [W, w]; % append parameters
         L(1,end+1) = l; % append current layer
         I(1:ng,end+1) = Gs(g,:)'; % append input index
         if ~any(Deltay), break, end % stop criterion: null error
      end
   end
   if ~any(Deltay), break, end % stop criterion: null error
   l = l + 1; % increment layer
end
model = struct('infer',@(x)infer(x,W,I,L,M,lb),'parameters',W(:,:).residue,...
   'inputs',I,'layers',L,'modulus',M,'depth',lb);
end



function y = infer(x,W,I,L,M,lb)
S = size(x,1); % number of samples
nl = max(L); % number of layers
nw = size(W,2); % number of parameter terms
X = {x}; for l = 0:nl, X{l+2} = rem(floor(x/2^(lb-l-1)),2); end
y = modular_residue(zeros(S,1),M); % output
for iw = 1:nw
   w = W(:,iw); while ~w(end), w(end) = []; end, % current parameters
   i = I(:,iw); while ~i(end), i(end) = []; end, ni = numel(i); % current inputs
   l = L(iw); % current layer
   xs = X{l+1}(:,i)*2.^(0:ni-1)'; % input groups
   V = modular_residue(cumprod([ones(S,1),repmat(xs,[1,S-1])],2),M); % Vandermonde matrix
   y = y + V*w; % reduce error
end
y = y(:).residue;
end