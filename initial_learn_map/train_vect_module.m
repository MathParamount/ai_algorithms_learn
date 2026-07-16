function [model, Deltay] = train_vect_module(xh, yh)

    if nargin < 2
        xh = [0 0; 0 1; 1 0; 1 1];
        yh = [0; 0; 0; 1];
    end

    n_entradas = size(xh, 2);
    S = size(xh, 1);
    
    %refinement method (2 inputs)
    if n_entradas <= 2
        
        if n_entradas == 1
            M = 2;
            l_bus = 0;
        else
            M = 5;
            l_bus = log2(M-1);  % = 2
        end

        if nargin < 3 || isempty(G)
            G = n_entradas + 1;
        end

        X = {xh};
        l = 1;
        Dy = yh;
        W = {};
        S_rand = randi([0 M-1], S, 1);
        L = zeros(1, 0);
        I = zeros(G, 0);

        while l <= numel(X) && ~isempty(X{l})
            N = size(X{l}, 2);

            if G <= N
                Gs = nchoosek(1:N, G);
            else
                Gs = 1:N;
            end

            % layer refinament (shift binário)
            if l <= l_bus
                X{l+1} = rem(floor(xh / (2^(l_bus-l-1))), 2);
            else
                X{l+1} = [];
            end

            for g = 1:size(Gs,1)
                xs = X{l}(:, Gs) * 2.^(0:length(Gs)-1)';

                if size(xs,1)==1, xs = xs'; end
                n = max(3, length(Gs));

                %vandermont matrix
                V = modular_residue(cumprod([ones(size(xs,1),1), repmat(xs,1,n-1)],2), M);
                v = double(V.residue);      %vandermont numerical value

                [~, ~, ju] = unique(xs);
                Dyp = accumarray(ju, Dy, [], @mean);
                Dyp = Dyp(ju);
                
                %
                if any(Dyp)
                    [rows_v, cols_v] = size(v);
                    rows_Dy = size(Dy,1);
                    if rows_Dy ~= rows_v
                        if rows_Dy > rows_v
                            Dy_adj = Dy(1:rows_v);
                        else
                            Dy_adj = [Dy; zeros(rows_v-rows_Dy,1)];
                        end
                    else
                        Dy_adj = Dy;
                    end

                    % increamental training (complete rank)
                    w = double(modular_residue(v \ Dy_adj, M).residue);
                    
                    %output error reduction
                    error = v * double(w);

                    % output error reduction
                    if size(error,1) < size(Dy,1)
                        error = [error; zeros(size(Dy,1)-size(error,1),1)];
                    elseif size(error,1) > size(Dy,1)
                        error = error(1:size(Dy,1));
                    end

                    % consistent error
                    Dy_temp = modular_residue(real(Dy - error), M);
                    Dy = double(Dy_temp.residue);

                    %append and change dimensionalities
                    W{end+1} = w;
                    L(1,end+1) = l;
                    I(1:size(Gs,2), end+1) = Gs(g,:)';

                    if ~any(Dy), break; end
                end
            end
            if ~any(Dy), break; end
            l = l + 1;
        end

        model.W = W;
        model.I = I;
        model.L = L;
        model.M = M;
        model.l_bus = l_bus;
        model.inference = @(x) inference_core(x, W, I, L, M, l_bus);
        Deltay = Dy;
    
    % polinomial method (3 inputs)
    else
        % nomial characteristics
        X_features = ones(S, 1);

        %keeping original features with news
        for i = 1:n_entradas
            X_features = [X_features, xh(:, i)];
        end

        %iterations among xh without pair repetition
        for i = 1:n_entradas
            for j = i+1:n_entradas
                X_features = [X_features, xh(:, i) .* xh(:, j)];
            end
        end

        %iterations with several variables
        if n_entradas >= 3
            X_features = [X_features, xh(:,1) .* xh(:,2) .* xh(:,3)];
        end

        %modulairy
        M = 3;
        
        %error adjust
        lambda = 1e-3;

        %weight
        w = mod((X_features'*X_features + lambda*eye(size(X_features,2))) \ (X_features'*yh), M);

        y_pred = mod(X_features * w, M);
        Deltay = mod(yh - y_pred, M);

        model.W = {w};
        model.I = (1:size(X_features,2))';
        model.L = 1;
        model.M = M;
        model.G = n_entradas + 1;
        model.l_bus = 0;
        model.inference = @(x) inference_poly(x, w, M, n_entradas);
    end
end


% polinomial inference
function y = inference_poly(x, w, M, n_entradas)

    if size(x,1)==1 && size(x,2)>1, x = x'; end

    S = size(x,1);          %sample
    X_features = ones(S,1);     %number of features
    
    %iterations with 1 variable
    for i = 1:n_entradas
        X_features = [X_features, x(:,i)];
    end
    
    %iteration with 2 variables
    for i = 1:n_entradas
        for j = i+1:n_entradas
            X_features = [X_features, x(:,i) .* x(:,j)];
        end
    end
    
    %multiple iterations
    if n_entradas >= 3
        X_features = [X_features, x(:,1) .* x(:,2) .* x(:,3)];
    end

    y_temp = modular_residue(round(X_features * w), M);
    y = double(y_temp.residue);
    y = mod(y, 2);   % binary normalization
end