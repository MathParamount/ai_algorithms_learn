function incremental_complex_train(xh,yh,G,G_1,N_1)

    %global variable
    D = 2;
    M = 20;
    N = 3;
    N_1 = 4; 


    %atribution input
    if nargin == 0
        xh = randi([0,M-1])^(N);
        yh = randi([0,M-1]);
        G = randi([2,floor(log2(M))]);
        G_1 = randi([2,floor(log2(M))])^(D);
        N_1 = prod( (0:(D-1)).* N_1);
    end

    Dy = yh;

    max_limit = log(max(N_1)) / log(G);
    
    %starting output error
    for r = 0:max_limit
        xh = mean(xh);
        
        for k = 0:(D-1)
            funct_1 = rem(i, prod ((0:k).* N_1));
            funct_2 = prod ( (0: (k-1).* N_1));
            funct_3 = (min(G_1,N_1)) / ( (1+e) * (N_1 - 1)); 
            
            arg = (funct_1 / funct_2);
            sub_arg = arg * funct_3 * prod( (0:(k-1)).* min(G_1,N_1) );
            i = sub_arg;
        end
        
        train_vect_module(xh, yh, G);
    end
end