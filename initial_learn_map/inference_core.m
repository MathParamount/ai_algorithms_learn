function y = inference_core(x,W,I,L,M,l_bar)
    
    addpath('/home/gabriel-cruz/projects/ai_algorithms_learn/initial_learn_map');

    if nargin < 6
        error('Inference should be call: model.inference(x)');
    end

    if nargin == 0
        x = [0; 1];
        W = [0; 1];
        S = numel(x);
        M = next_prime(max(max(x(:))+1,S)); % modulus
        
        maxG = max(2, floor(log2(M))); 
        G = randi([2, maxG]);

        I = zeros(G,0); % parameter input
        L = zeros(1,0); % parameter layer
        l_bar = log2(M-1); % maximum layer depth
    end

    S = size(x,1); % number of samples
    
    % number of layers
    if isempty(L)
        nl = 0;
    else
        nl = max(L);
    end
    
    % number of parameter terms
    nw = size(W,2);

    X = cell(nl+1,1);

    X{1}=x;

    for l=2:nl+1

        previous = X{l-1};

        X{l}=mod(previous(:,1:end-1)+previous(:,2:end),M);
    end

    y = modular_residue(zeros(S,1),M); % output

    for iw = 1:nw
        w = W(:,iw);
           
        %remove zeros
        idx = find(w~=0);

        if isempty(idx)
            continue
        end

        w = w(1:idx(end));

        % current parameters
        i = I(:,iw);
        i = i(i~=0); 

        while ~isempty(i) && ~i(end)
            i(end) = []; 
        end
        
        ni = numel(i); % current inputs

        if ni == 0
            continue; % sem entradas, pula termo
        end
        
        layer = L(iw);      %current layer

        if layer+1 > numel(X)
            continue
        end
        
        x_layer = X{layer+1};

        if i > size(x_layer,2)
            xs = x_layer(:,end);
        else
            xs = x_layer(:,i);
        end

        %inputs combination
        if size(xs,2)==1
            z = xs;
        else
            z = sum(xs,2);
        end

        z = mod(z,M);

        v = [
            ones(S,1), ...
            xs(:,1), ...
            xs(:,end), ...
            mod(xs(:,1).* xs(:,end),M);
        ];

        y = mod(y + v*w, M);
    end
end


%module residue inplementation
function r = modular_residue(a, M)
    % Reduz a matriz/vetor 'a' módulo M
    r = mod(a, M);
end