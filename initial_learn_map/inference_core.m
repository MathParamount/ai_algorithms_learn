function y = inference_core(x,W,I,L,M,l_bar)
    
    addpath('/home/gabriel-cruz/projects/ai_algorithms_learn/initial_learn_map');

    if size(x,1) == 1 && size(x,2) > 1
        x = x';
    end
    
    S = size(x, 1);      % number of features (features in cols)
    y = zeros(S,1);
    
    %lack of weight
    if isempty(W)
        warning('No weight available. Returning zeros');        
        return;
    end

    X = {x};
    nl = max(L);

    %refined layers (superior layer (l+2) )
    for l= 0:nl
        if l <= l_bar
            X{l+2} = rem(floor(x / (2^(l_bar-l-1))),2);
        else
            X{l+2} = [];
        end
    end

    y_pred = zeros(S,1);    %acumule predictions

    % weight tratment
    for iw = 1:length(W)
            w = W{iw};

            if size(w,1) == 1
                w = w';
            end
            
            if iw <= size(I, 2)
                Gs = I(:, iw);
                Gs = Gs(Gs ~= 0);
            else
                Gs = 1:min(length(w)-1, size(x, 2));
            end

           
            %Layer length verification
            if iw <= length(L)
                l = L(iw); % current layer
            else
                l = 1;
            end

            %refined layer verification
            X_layer = X{l};

            X_selecionado = X_layer(:, Gs);
        
            % Combining inputs (input groups)
            xs = X_selecionado * 2.^(0:length(Gs)-1)';
        
            if size(xs, 1) == 1
                xs = xs';
            end
            
            %highest order
            n = length(w);
        
            V = modular_residue(cumprod([ones(S, 1), repmat(xs, 1, n-1)], 2), M);
            v = double(V.residue);
        
            %adjusting to w length
            if size(v, 2) > length(w)
                v = v(:, 1:length(w));
            elseif size(v, 2) < length(w)
                v = [v, zeros(size(v, 1), length(w) - size(v, 2))];
            end
            
            %prediction
            y_pred = y_pred + v * double(w);

        end
        
        y_temp = modular_residue(round(y_pred), M);
        y = double(y_temp.residue);
end