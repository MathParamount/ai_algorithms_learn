% xh - modulo M input
% yh - binary output
% G - input group size
% 
%  bmai([0; 1],[1; 0])
%
% compilation test:
%
% clc; clear;
% xh = [ 0 0; 0 1; 1 0; 1 1];
% yh = [0; 0; 0; 1];
% [model, Dy] = train_vect_module(xh, yh);
% disp('train executed without error');
% disp('Dy final:');
% disp(Dy);
% ------------------
% Inference test:
%
% clc; clear;
% xh = [ 0 0; 0 1; 1 0; 1 1];
% yh = [0; 0; 0; 1];
% [model,~] = train_vect_module(xh, yh);
% y = model.inference(xh');
% disp('model output: ');
% disp(y);
% ------------------
% Prediction and inference test:
%
% clc; clear;
% xh = [0 0; 0 1; 1 0; 1 1];
% yh = [0;0;0;1];
% [model, ~] = train_vect_module(xh, yh);
% y = model.inference(xh');
% expected = yh;
% disp('prediction:'); disp(y);
% disp('expected:'); disp(expected);
% yv = y(:); ev = expected(:);
% n = min(numel(yv), numel(ev));
% error = sum(abs(yv(1:n) - ev(1:n)));
% disp('total error:');
% disp(error);

%-------------------------------------
% FALSE

% xh = [0 0 0; 0 0 1; 0 1 0;0 1 1; 1 0 0; 1 0 1; 1 1 0; 1 1 1];
% yh = [0; 0; 0; 0; 0; 0; 0; 0];
% [model, ~] = train_vect_module(xh, yh);
% y = model.inference(xh');
% expected = yh;
% disp('prediction:'); disp(y);
% disp('expected:'); disp(expected);
% yv = y(:); ev = expected(:);
% n = min(numel(yv), numel(ev));
% error = sum(abs(yv(1:n) - ev(1:n)));
% disp('total error:');
% disp(error);

% TRUE

% xh = [0 0 0; 0 0 1; 0 1 0;0 1 1; 1 0 0; 1 0 1; 1 1 0; 1 1 1];
% yh = [0; 1; 1; 0; 1; 0; 0; 1];
% [model, ~] = train_vect_module(xh, yh);
% y = model.inference(xh');
% expected = yh;
% disp('prediction:'); disp(y);
% disp('expected:'); disp(expected);
% yv = y(:); ev = expected(:);
% n = min(numel(yv), numel(ev));
% error = sum(abs(yv(1:n) - ev(1:n)));
% disp('total error:');
% disp(error);

% NAND

% xh = [0 0 0; 0 0 1; 0 1 0;0 1 1; 1 0 0; 1 0 1; 1 1 0; 1 1 1];
% yh = [1; 1; 1; 1; 1; 1; 1; 0];
% [model, ~] = train_vect_module(xh, yh);
% y = model.inference(xh');
% expected = yh;
% disp('prediction:'); disp(y);
% disp('expected:'); disp(expected);
% yv = y(:); ev = expected(:);
% n = min(numel(yv), numel(ev));
% error = sum(abs(yv(1:n) - ev(1:n)));
% disp('total error:');
% disp(error);

% NOR
% xh = [0 0 0; 0 0 1; 0 1 0;0 1 1; 1 0 0; 1 0 1; 1 1 0; 1 1 1];
% yh = [1; 0; 0; 0; 0; 0; 0; 0];
% [model, ~] = train_vect_module(xh, yh);
% y = model.inference(xh');
% expected = yh;
% disp('prediction:'); disp(y);
% disp('expected:'); disp(expected);
% yv = y(:); ev = expected(:);
% n = min(numel(yv), numel(ev));
% error = sum(abs(yv(1:n) - ev(1:n)));
% disp('total error:');
% disp(error);


% XNOR

%xh = [0 0 0; 0 0 1; 0 1 0;0 1 1; 1 0 0; 1 0 1; 1 1 0; 1 1 1];
% yh = [1; 0; 0; 1; 0; 1; 1; 0];
% [model, ~] = train_vect_module(xh, yh);
% y = model.inference(xh');
% expected = yh;
% disp('prediction:'); disp(y);
% disp('expected:'); disp(expected);
% yv = y(:); ev = expected(:);
% n = min(numel(yv), numel(ev));
% error = sum(abs(yv(1:n) - ev(1:n)));
% disp('total error:');
% disp(error);


% AND
% xh = [0 0; 0 1; 1 0; 1 1]; yh = [0; 0; 0; 1]; % dataset
% [model,Deltay] = mai(xh,yh) % training
% y = model.infer(xh) % inference

% XOR
% Dataset de paridade (XOR de 3 entradas)
% xh = [0 0 0; 0 0 1; 0 1 0; 0 1 1; 1 0 0; 1 0 1; 1 1 0; 1 1 1];
% yh = [0; 1; 1; 0; 1; 0; 0; 1];
% [model, Deltay] = train_vect_module(xh, yh);
%y = model.inference(xh);



%###########Measure of train###############
%expected = yh;
%disp('Prediction:'); disp(y);
%disp('Expected:'); disp(expected);

% Calcula erro
%error = sum(abs(y - expected));
%disp('Total error:');
%disp(error);

% Acurácia
%accuracy = sum(y == expected) / length(expected) * 100;
%disp(['Accuracy: ', num2str(accuracy), '%']);