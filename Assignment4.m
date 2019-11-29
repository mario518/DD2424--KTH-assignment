clear
clc

book_fname = 'data/goblet_book.txt';
fid = fopen(book_fname,'r');
book_data = fscanf(fid,'%c');
fclose(fid);


book_chars = unique(book_data);

char_to_ind = containers.Map('KeyType','char','ValueType','int32');
ind_to_char = containers.Map('KeyType','int32','ValueType','char');


key = num2cell(book_chars);
value = 1 : length(key);
Map1 = containers.Map(key, value);
Map2 = containers.Map(value, key);
char_to_ind = [char_to_ind; Map1];
ind_to_char = [ind_to_char; Map2];

K = length(key);             
m = 100;                     
eta = 0.1;                      
seq_length = 25;            
sig = 0.01;     

RNN.b = zeros(m, 1);              
RNN.c = zeros(K, 1);
RNN.U = randn(m, K)*sig;       
RNN.W = randn(m, m)*sig;
RNN.V = randn(K, m)*sig;

M.W = zeros(size(RNN.W));
M.U = zeros(size(RNN.U));
M.V = zeros(size(RNN.V));
M.b = zeros(size(RNN.b));
M.c = zeros(size(RNN.c));

% %% Check gradients
% X_ind = zeros(1, length(book_data));
% for i = 1 : length(book_data)        
%     X_ind(i) = char_to_ind(book_data(i));
% end
% X = oneHot(X_ind, K);
% batch_size = 25;
% dh = 1e-4;
% gradCheck(batch_size, X(:, 1 : seq_length), X(:, 2 : seq_length + 1), RNN, dh, m, K);

%%  training process using AdaGrad

    X_ind = zeros(1, length(book_data));
    for i = 1 : length(book_data)
        X_ind(i) = char_to_ind(book_data(i));
    end
    X = oneHot(X_ind, K);

Y = X;     
iter = 1;
n_epochs = 9;
SL = [];
sl = 0;
hprev = [];
min_loss = 500;
for i = 1 : n_epochs
    [RNN, sl, iter, M, min_RNN, min_h, min_iter, min_loss] = MiniBatchGD(RNN, X, Y, seq_length, K, m, eta, iter, M, ind_to_char, sl(end), min_loss);
    SL = [SL, sl];
end


textlen = 1000;
min_y  = [];
 y = synText(min_RNN, min_h, X(:, 1), textlen, K);
 text = [];
 min_y = [min_y y];
 for i = 1 : textlen
     text = [text ind_to_char(min_y(i))];
 end

disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
disp(['min_iter = ' num2str(min_iter) ', min_loss = ' num2str(min_loss)]);
disp(text);

figure()
plot(1 : length(SL), SL);
title('Smooth Loss VS Iterations');
xlabel('Iterations');
ylabel('Smooth Loss');

%% Synthesize Text
function y = synText(RNN, h0, x0, n, K)
% initialize parameters
W = RNN.W;
U = RNN.U;
V = RNN.V;
b = RNN.b;
c = RNN.c;
h = h0;
x = x0;
y = zeros(1, n);

for t = 1 : n
    a = W*h + U*x + b;
    h = tanh(a);
    o = V*h + c;
    p =exp(o)/sum(exp(o));
    
    % randomly select a character based on probability
    cp = cumsum(p);
    a = rand;
    ixs = find(cp - a >0);
    ii = ixs(1);
    
    % generate the next input
    x = oneHot(ii, K);
    y(t) = ii;
end
    
end

%% One hot
function out = oneHot(label, K)

N = length(label);
out = zeros(K, N);

for i = 1 : N
    out(label(i), i) = 1;
end

end


%% Forward Pass
    
function [loss, a, h, o, p] = forward_Pass(RNN, X, Y, h0, n, K, m)
% initialize parameters
W = RNN.W;
U = RNN.U;
V = RNN.V;
b = RNN.b;
c = RNN.c;
ht = h0;

a = zeros(m, n);
h = zeros(m, n);
o = zeros(K, n);
p = zeros(K, n);


loss = 0;

for t = 1 : n
    a(:, t) = W*ht + U*X(:, t) + b;  
    ht = tanh(a(:,t));
    h(:, t) = ht;
    o(:, t) = V*ht + c;
    p(:, t) = exp(o(:, t))/sum(exp(o(:, t)));
    loss = loss - log(Y(:, t)'*p(:, t));
end

h = [h0, h];

end

%%  Compute Gradients
function grads = ComputeGradients(RNN, X, Y, a, h, p, n, m)
% initialize parameters
W = RNN.W;
V = RNN.V;
g_h = zeros(n, m);
g_a = zeros(n, m);

g = -(Y - p)';                                    
grads.c = (sum(g))';                                    
grads.V = g'*h(:, 2 : end)';                       

g_h(n, :) = g(n, :)*V;                                
g_a(n, :) = g_h(n, :)*diag(1 - (tanh(a(:, n))).^2);     

for t = n - 1 : -1 : 1
    g_h(t, :) = g(t, :)*V + g_a(t + 1, :)*W;
    g_a(t, :) = g_h(t, :)*diag(1 - (tanh(a(:, t))).^2);
end

grads.b = (sum(g_a))';                              
grads.W = g_a'*h(:, 1 : end - 1)';               
grads.U = g_a'*X';                                    

end

%% Backward Pass
function [RNN, M] = backward_Pass(RNN, X, Y, a, h, p, n, m, eta, M)

grads = ComputeGradients(RNN, X, Y, a, h, p, n, m);
eps = 1e-8;

for f = fieldnames(RNN)'
    
    grads.(f{1}) = max(min(grads.(f{1}), 5), -5);

    M.(f{1}) = M.(f{1}) + grads.(f{1}).^2;
    RNN.(f{1}) = RNN.(f{1}) - eta*(grads.(f{1})./(M.(f{1}) + eps).^(0.5));
end

end

%% Gradient Check
function gradCheck(batch_size, X, Y, RNN, dh, m, K)

% numerical gradients
n_grads = ComputeGradsNum(X(:, 1 : batch_size), Y(:, 1 : batch_size), RNN, dh);

% analytical gradients
h0 = zeros(size(RNN.W, 1), 1);
[~, a, h, ~, p] = forward_Pass(RNN, X(:, 1 : batch_size), Y(:, 1 : batch_size), ...
    h0, batch_size, K, m);
grads = ComputeGradients(RNN, X(:, 1 : batch_size), Y(:, 1 : batch_size), ...
    a, h, p, batch_size, m);

eps = 1e-5;

for f = fieldnames(RNN)'
    num_g = n_grads.(f{1});
    ana_g = grads.(f{1});
    denominator = abs(num_g) + abs(ana_g);
    numerator = abs(num_g - ana_g);
    gradcheck_max = max(numerator(:))/max(eps, sum(denominator(:)));
    disp(['Field name: ' f{1}]);
    disp(['max error: ' num2str(gradcheck_max) ]);
end

end


%% Compute Loss
function loss = ComputeLoss(X, Y, RNN, h)
W = RNN.W;
U = RNN.U;
V = RNN.V;
b = RNN.b;
c = RNN.c;
n = size(X, 2);
loss = 0;

for t = 1 : n
    at = W*h + U*X(:, t) + b;
    h = tanh(at);
    o = V*h + c;
    pt = exp(o);
    p = pt/sum(pt);

    loss = loss - log(Y(:, t)'*p);
end

end

%% MiniBatchGD
function [RNN, sl, iter, M, min_RNN, min_h, min_iter, min_loss] = MiniBatchGD(RNN, X, Y, n, K, m, eta, iter, M, ind_to_char, smooth_loss, min_loss)

e = 1;
textlen = 1000;
sl = [];
while e <= length(X) - n - 1
    Xe = X(:, e : e + n - 1);
    Ye = Y(:, e + 1 : e + n);
    if e == 1
        hprev = zeros(m, 1);
    else
        hprev = h(:, end);
    end
    
    [loss, a, h, ~, p] = forward_Pass(RNN, Xe, Ye, hprev, n, K, m);
    [RNN, M] = backward_Pass(RNN, Xe, Ye, a, h, p, n, m, eta, M);
    
    if iter == 1 && e == 1
        smooth_loss = loss;
    end
    smooth_loss = 0.999*smooth_loss + 0.001*loss;
    if smooth_loss < min_loss
        min_RNN = RNN;
        min_h = hprev;
        min_iter = iter;
        min_loss = smooth_loss;
    end
    sl = [sl, smooth_loss];
    
    if iter == 1 || mod(iter, 500) == 0
        y = synText(RNN, hprev, X(:, 1), textlen, K);
        c = [];
        for i = 1 : textlen
            c = [c ind_to_char(y(i))];
        end
        disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
        disp(['iter = ' num2str(iter) ', smooth_loss = ' num2str(smooth_loss)]);
        disp(c);
    end
    
    iter = iter + 1;
    e = e + n;
end

end

%% ComputeGradsNum
function num_grads = ComputeGradsNum(X, Y, RNN, h)

for f = fieldnames(RNN)'

    num_grads.(f{1}) = ComputeGradNumSlow(X, Y, f{1}, RNN, h);
end
end

function grad = ComputeGradNumSlow(X, Y, f, RNN, h)

n = numel(RNN.(f));
grad = zeros(size(RNN.(f)));
hprev = zeros(size(RNN.W, 1), 1);
for i=1:n
    RNN_try = RNN;
    RNN_try.(f)(i) = RNN.(f)(i) - h;
    l1 = ComputeLoss(X, Y, RNN_try, hprev);
    RNN_try.(f)(i) = RNN.(f)(i) + h;
    l2 = ComputeLoss(X, Y, RNN_try, hprev);
    grad(i) = (l2-l1)/(2*h);
end
end