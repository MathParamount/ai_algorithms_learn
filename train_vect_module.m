function train_vect_module(xh,yh,G)
% xh - modulo M input
% yh - binary output
% G - input group size
% 
%  bmai([0; 1],[1; 0])

if nargin < 3
    xh = [0; 1];
    yh = [0; 1];
end


% cardinality
M = max(xh(:))+1; % modulus

% input arguments
if nargin < 3
    maxG = max(2, floor(log2(M))); 
    G = randi([2, maxG]);
end

% initialization
X = {xh}; % starting input set
l = 1; % layer counter
l_bus = log2(M-1); % maximum layer depth
Dy = yh; % starting output error
W = []; % starting paramter set
s = randi(10);

while l <= numel(X) && ~isempty(X{l})
    % input groups
    N = size(X{l},2);
    e = randi(s-1);
   
    if G < N
        %combinatorial operation
        Gs = nchoosek(1:N,G);
    else
        Gs = 0:N;
    end
    
    g_bus = size(Gs,1);

    %refined input
    if l < l_bus && N > 1
        x_l_1 = rem(xh*G,M);
    else
        x_l_1 = {};
    end
   
    %input ranges
    for g = 1:size(Gs,1) 
        if g >= rem(M,g_bus)
            x_g = M/g_bus + 0;
        else
            x_g = M/g_bus + 1;
        end

        % input grouping
        S = sum( (xh/(M-1)) * (x_g(1:g)/(1+e)) );
        P = prod(x_g(1:g));
        xh_1 = S*P;

        %vandermonde matrix
        v = rem(xh_1,M);

        %consistent error,skip training
        Dy = min(Dy);

        %incremental weight
        w = rem(inv(v) * Dy, M);

        %output error reduction
        Dy = Dy - v*w;

        %append parameters
        W = union(W, w);
    end
    
    l = l + 1;     %incremental layer

 end

end