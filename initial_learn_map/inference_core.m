function y = inference_core(x, W, I, L, M, l_bar)

    % data prepartion
    if size(x, 1) == 1 && size(x, 2) > 1
        x = x';
    end
    
    S = size(x, 1);          % sample number
    y = zeros(S, 1);
    
    if isempty(W)
        warning('Nenhum peso disponível. Retornando zeros.');
        return;
    end

    X = {x};                 % original input
    nl = max(L);             %maximum number of layer

    for l = 0:nl
        if l <= l_bar
            % ancient layer(X{l+1})
            Xprev = X{l+1};
            Xnew = zeros(size(Xprev));
            
            for k = 1:size(Xprev, 2)
                Xnew(:, k) = mod(3 * Xprev(:, k) + k + (l + 1), M);
            end
            X{l+2} = Xnew;
        else
            X{l+2} = [];
        end
    end

    % inference
    y_pred = zeros(S, 1);

    for iw = 1:length(W)
        w = W{iw};
        if size(w, 1) == 1
            w = w';
        end

        % Recover input index
        if iw <= size(I, 2)
            Gs = I(:, iw);
            Gs = Gs(Gs ~= 0);
        else
            Gs = 1:min(length(w) - 1, size(x, 2));
        end

        % actual layer
        if iw <= length(L)
            l = L(iw);
        else
            l = 1;
        end

        % select correspondend layer
        X_layer = X{l};
        X_selecionado = X_layer(:, Gs);

        %combine input selected with input
        xs = X_selecionado * 2.^(0:length(Gs) - 1)';
        if size(xs, 1) == 1
            xs = xs';
        end

        % vandermont order
        n = max(4, length(Gs));

        % vandermont matrix
        V = modular_residue(cumprod([ones(S, 1), repmat(xs, 1, n - 1)], 2), M);
        v = double(V.residue);

        % adjusting v to w length
        if size(V, 2) > length(w)
            v = v(:, 1:length(w));
        elseif size(V, 2) < length(w)
            v = [v, zeros(size(v, 1), length(w) - size(v, 2))];
        end

        % contribution accumulation
        y_pred = y_pred + v * double(w);
    end

    %apply module and binary normalization
    y_temp = modular_residue(round(y_pred), M);
    y = double(y_temp.residue);

    if M > 2
        y = mod(y, 2);
    end
end