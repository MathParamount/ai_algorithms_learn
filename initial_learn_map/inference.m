function y = inference(x,W,I,L,M,l_bar)
    
    if nargin == 0
        x = [0; 1];
        W = [0; 1];
        S = numel(x);
        M =  next_prime(max(max(x(:))+1,S)); % modulus
        
        maxG = max(2, floor(log2(M))); 
        G = randi([2, maxG]);

        I = zeros(G,0); % parameter input
        L = zeros(1,0); % parameter layer
        l_bar = log2(M-1); % maximum layer depth
    end

    S = size(x,1); % number of samples
    nl = max(L); % number of layers
    nw = size(W,2); % number of parameter terms
    X = {x}; 
    
    for l = 0:nl
        X{l+2} = rem(floor(x/2^(l_bar-l-1)),2); 
    end
    
    y = modular_residue(zeros(S,1),M); % output

    for iw = 1:nw
        w = W(:,iw);
           
        %remove zeros
        while ~w(end) && ~isempty(w)
            w(end) = []; 
        end

        if isempty(w)
            continue;
        end

        % current parameters
        i = I(:,iw); 
        while ~i(end) && isempty(i)
            i(end) = []; 
        end
        
        ni = numel(i); % current inputs

        if ni == 0
            continue; % sem entradas, pula termo
        end
        
        l = L(iw); % current layer
        
        K = length(w);
        vand_matrix = ones(S, K);
        if K > 1
            vand_matrix(:,2:K) = cumprod(repmat(xs, 1, K-1), 2);
        end

        vand_matrix = modular_residue(vand_matrix, M);
        
        y = y + vand_matrix*w; % reduce error
    end
    y = y(:).residue;
end