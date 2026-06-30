function fixed_modl_train(xh,yh, G, G_1, N_1)
    
    %global variables (default adjust)
    M = 20;
    D = 2;
    erro = 0.05;
    N_1 = 4; 
    
    %verification to input empty
    if nargin == 0
        xh = randi([0,M-1]);
        yh = randi([0,M-1]);
        G = randi([2,floor(log2(M))]);
        G_1 = randi([2,floor(log2(M))])^(D);
        
        seq = (0:D-1).*N_1;
        N = prod(seq);
    end
    
    Dy = yh;
    i = 10; 

    oper = log(max(N_1)) / log(G);
    
    %to fix with decimal type 
    oper_limite = floor(oper); 

    %refined input
    I = 0;


    for r = 0:oper_limite
        xh_1 = mean(xh);

        %3: starting output error
        for  k= 1: (D-1)    %summer limits
            
            term1 = prod( (1:k).*N_1 );
            term2 = prod((1:max(1, k-1)) .* N_1);
            term3 = prod ( (1:k).* min(G_1, N_1));

            sum1 = rem(i, term1) / term2;
            sum2 = sum1 * min(G_1, N_1) / ((1 + erro) * (N_1 - 1));
            sum3 = sum1 * sum2 * term3;

            I = I + sum3;
        end

       train_vect_module(xh, yh, G);
    end
    
end