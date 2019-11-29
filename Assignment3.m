addpath Datasets\cifar-10-batches-mat;
clear
clc

%%%%%%%%%%%%%%%%%%
%% Without Batch Normalization
%%%%%%%%%%%%%%%%%%


%% Load Data
[train_X1,train_Y1,train_y1]=LoadBatch('data_batch_1.mat');
[train_X2,train_Y2,train_y2]=LoadBatch('data_batch_2.mat');
[train_X3,train_Y3,train_y3]=LoadBatch('data_batch_3.mat');
[train_X4,train_Y4,train_y4]=LoadBatch('data_batch_4.mat');
[train_X5,train_Y5,train_y5]=LoadBatch('data_batch_5.mat');
% % 
train_X=[train_X1,train_X2(:,5001:size(train_X2,2)),train_X3, train_X4, train_X5];
train_Y=[train_Y1,train_Y2(:,5001:size(train_X2,2)),train_Y3, train_Y4, train_Y5];
train_y=[train_y1,train_y2(:,5001:size(train_X2,2)),train_y3, train_y4, train_y5];

validation_X=train_X2(:,1:5000);
validation_Y=train_Y2(:,1:5000);
validation_y=train_y2(:,1:5000);

[testX,testY,testy]=LoadBatch('test_batch.mat');

% [train_X,train_Y,train_y]=LoadBatch('data_batch_1.mat');
% [validation_X,validation_Y,validation_y]=LoadBatch('data_batch_2.mat');
% [testX,testY,testy]=LoadBatch('test_batch.mat');

mean_X = mean (train_X,  2);
std_X = std(train_X, 0, 2);
train_X = train_X - repmat(mean_X,[1, size(train_X,2)]);
train_X = train_X./ repmat(std_X,[1, size(train_X,2)]);

validation_X = validation_X - repmat(mean_X,[1, size(validation_X,2)]);
validation_X = validation_X./ repmat(std_X,[1, size(validation_X,2)]);

testX = testX - repmat(mean_X,[1, size(testX,2)]);
testX = testX./ repmat(std_X,[1, size(testX,2)]);

%% Initialize Parameters
m= [50,50];
batch_size = 100;
k = size(m,2)+1;
% lambda = 0.005;
lambda = 0.005;
delta = 1e-6;
dimension = 10;

%% Check gradients
% gradCheck(dimension,batch_size, train_X, train_Y, lambda, delta, k,m)

%% Plot and Accuracy Check
n_batch = 100;
LR.eta_min = 1e-5;
LR.eta_max = 1e-1;
LR.n_s = 5* 45000/n_batch;
LR.t =4 *LR.n_s;
eta_t =  LearningRate(LR.eta_min, LR.eta_max, LR.n_s,LR.t);
% LR.n_s = 500;

N =size(train_X,2);
J_validation = zeros(1,LR.t/N*n_batch);
J_train = zeros(1,LR.t/N*n_batch);
accuracy_train = zeros(1,LR.t/N*n_batch);
accuracy_validation = zeros(1,LR.t/N*n_batch);
[W, b] = Initializing_network(train_X,m);
count = 0;

for e=1:LR.t/N*n_batch
    tic;
        J_train(e) = ComputeCost(train_X, train_Y, W, b, lambda,k);
        J_validation(e) = ComputeCost(validation_X, validation_Y, W, b, lambda,k);
        
        accuracy_train(e) = ComputeAccuracy(train_X,train_y,W,b,k);
        accuracy_validation(e) = ComputeAccuracy(validation_X,validation_y,W,b,k);
        [Wstar,bstar] = MiniBatchGD(train_X,train_Y, W,b,N,k,count,n_batch, eta_t,lambda);
        W=Wstar;
        b=bstar;
        count = count+1;
            
%         shuffle data
        index = randperm(size(train_X,2));
        train_X = train_X(:,index);
        train_Y = train_Y(:,index);
        train_y = train_y(:,index);
     toc
end 

figure()
plot (1:LR.t/N*batch_size, J_train, 'g')
hold on
plot (1:LR.t/N*batch_size, J_validation, 'r')
hold off
legend('training cost','validation cost')
legend('training cost')
xlabel('epochs')
ylabel('cost')
% xlabel('update step')


figure()
plot (1:LR.t/N*batch_size, accuracy_train, 'b')
hold on
plot (1:LR.t/N*batch_size, accuracy_validation, 'r')
hold off
legend('training accuracy','validation accuracy')
xlabel('epochs')
ylabel('accuracy')

disp(['lambda:' num2str(lambda)])
accuracy_train = ComputeAccuracy(train_X,train_y,W,b,k);
disp(['Training Accuracy:' num2str(accuracy_train*100) '%'])

accuracy_test = ComputeAccuracy(testX,testy,W,b,k);
disp(['Test Accuracy:' num2str(accuracy_test*100) '%'])


%%
function gradCheck(dimension,batch_size, X, Y, lambda, delta, k,m)
[W, b] = Initializing_network(X(1:dimension, 1 : batch_size),m);
% numerical gradients
[ngrad_W,ngrad_b] = ComputeGradsNumSlow(X(1:dimension, 1 : batch_size), ...
    Y(1:dimension, 1 : batch_size), W, b, lambda, delta,k);

% analytical gradients
[h, s] = intervalues(X(1:dimension, 1 : batch_size), W, b, k);
P = EvaluateClassifier(h, W, b);
[grad_W, grad_b] = ComputeGradients(X(1:dimension, 1 : batch_size), ...
    Y(1:dimension, 1 : batch_size),  W, b, k, lambda);

% relative error rate
eps = 1e-10;
for i = 1 : length(W)
    gradcheck_bi = sum(abs(ngrad_b{i} - grad_b{i})/max(eps, sum(abs(ngrad_b{i}) + abs(grad_b{i}))));
    gradcheck_bm = max(abs(ngrad_b{i} - grad_b{i})/max(eps, sum(abs(ngrad_b{i}) + abs(grad_b{i}))));
    gradcheck_Wi = sum(sum(abs(ngrad_W{i} - grad_W{i})/max(eps, sum(sum(abs(ngrad_W{i}) + abs(grad_W{i}))))));
    gradcheck_Wm = max(max(abs(ngrad_W{i} - grad_W{i})/max(eps, sum(sum(abs(ngrad_W{i}) + abs(grad_W{i}))))));
    disp(['error of grad_W' num2str(i) ': ' num2str(gradcheck_Wi)])
    disp(['max error of grad_W' num2str(i) ': ' num2str(gradcheck_Wm)])
    disp(['error of grad_b' num2str(i) ': ' num2str(gradcheck_bi)])
    disp(['max error of grad_b' num2str(i) ': ' num2str(gradcheck_bm)])
end

end


%% load the data
function [X, Y, y] = LoadBatch(filename)
Dataset = load(filename);
data = Dataset.data';
label = Dataset.labels;
newlabel = label + 1;

Y_array = zeros(10,10000);

for i = 1:10000
    Y_array(newlabel(i),i)=1;
end

X = double(data)/255;
Y = double(Y_array);
y = double(newlabel');
end

% initialize the parameters
%%
function [W,b] = Initializing_network(X,m)
d = size(X,1);
% sig=1e-4;
% W{1} = sig.*randn(m(1), d);
W{1} = 1/sqrt(d).*randn(m(1), d);
b{1} = zeros(m(1),1);
for i = 1:size(m,2)
     %Xavier initialization
%     std = sig;
    std = sqrt(1/m(i));
 if i~=size(m,2)
    W{i+1} = std.* randn (m(i+1),m(i));
    b{i+1} = zeros(m(i+1),1);
 else    W{i+1} = std.*randn(10,m(end));
    b{i+1} = zeros(10,1);
 end
end 
end

%%
function [h,s]=intervalues(X,W,b,k)
s{1} = W{1}*X+b{1};
x{1} = max(0, s{1});
h{1} = x {1};
for i = 2:k-1
    s{i} = W{i}*x{i-1}+b{i};
    x{i} = max(0,s{i});
    h{i} = x{i};
end
s{k} = W{k}*x{k-1}+b{k};
end 



%%
function P = EvaluateClassifier(h, W, b)

W = W{end};
b = b{end};
X = h{end};
b = repmat(b, 1, size(X, 2));

s = W*X + b;
denorm = repmat(sum(exp(s), 1), size(s, 1), 1);
P = exp(s)./denorm;

end


%%
function J = ComputeCost(X, Y, W, b, lambda, k)

 [h,~] = intervalues(X, W, b, k);

P = EvaluateClassifier(h, W, b);

    sumP = 0;
    J2=0;
    for i = 1:size(Y,2)
        sumP =   sumP + (-log(Y(:,i)'*P(:,i)));
    end
    J1 = 1/size(X,2)*sumP ;
    
for i = 1 : length(W)
    temp = W{i}.^2;
    J2 = J2 + lambda*sum(temp(:));
end

    J = J1+ J2;
end


%%
function acc = ComputeAccuracy(X, y, W, b,k)

[h,~]=intervalues(X,W,b,k);
P = EvaluateClassifier(h, W, b);
[~, ind] = max(P);
acc = length(find(y - ind == 0))/length(y);
end


%%  
function [grad_W, grad_b] = ComputeGradients(X, Y, W, b, P,h,k,lambda)


G = -(Y - P);
for l = k:-1:2
    grad_W{l}= G*h{l-1}'/size(X,2)+2* lambda*W{l};
    grad_b{l} = G* ones(size(X,2),1)/size(X,2);
    G = W{l}'* G;
    Ind = h{l -1}>0;
    G = G.* Ind;
end
grad_b{1} = G* ones(size(X,2),1)/size(X,2);
grad_W{1}= G * X'/size(X,2)+2* lambda*W{1};
end



%% [Wstar, bstar] = MiniBatchGD(trainX, trainY, GDparams, W, b, lambda);
 function [Wstar,bstar] = MiniBatchGD(X,Y, W,b,N,k,count,n_batch, eta_t,lambda)


for j=1:N/n_batch
        j_start = (j-1)*n_batch + 1;
        j_end = j*n_batch;
        inds = j_start:j_end;
        Xbatch = X(:, inds);
        Ybatch = Y(:, inds);
        index = j + count* N/n_batch;
        
        
        [h,~]=intervalues(Xbatch,W,b,k);
        P = EvaluateClassifier(h, W, b);
     
        [grad_W, grad_b] = ComputeGradients(Xbatch, Ybatch,W, b,P,h, k,lambda);
        for l = 1:k
        W{l}= W{l} - eta_t(index) .* grad_W{l};
        b{l} = b{l} - eta_t (index).* grad_b{l};  
        end        
end
Wstar = W;
bstar = b;
end 

%% ComputeGradsNumSlow
function [grad_W, grad_b] = ComputeGradsNumSlow(X, Y, W, b, lambda, h,k)

grad_W = cell(1,numel(W));
grad_b = cell(1,numel(b));

for j=1:length(b)
    grad_b{j} = zeros(size(b{j}));
    
    for i=1:length(b{j})
        
        b_try = b;
        b_try{j}(i) = b_try{j}(i) - h;
        c1 = ComputeCost(X, Y, W, b_try, lambda,k);
        
        b_try = b;
        b_try{j}(i) = b_try{j}(i) + h;
        c2 = ComputeCost(X, Y, W, b_try, lambda,k);
        
        grad_b{j}(i) = (c2-c1) / (2*h);
    end
end

for j=1:length(W)
    grad_W{j} = zeros(size(W{j}));
    
    for i=1:numel(W{j})
        
        W_try = W;
        W_try{j}(i) = W_try{j}(i) - h;
        c1 = ComputeCost(X, Y, W_try, b, lambda,k);
    
        W_try = W;
        W_try{j}(i) = W_try{j}(i) + h;
        c2 = ComputeCost(X, Y, W_try, b, lambda,k);
    
        grad_W{j}(i) = (c2-c1) / (2*h);
   end
end
end 

%% Learning rate
function eta_t = LearningRate(eta_min, eta_max, n_s,t)
eta_t= zeros(1,t);

for j = 1: t
    term_1 = 0;
    term_2 = 0;
    for i= 0:5
     term1 = (eta_min +( j-2*i*n_s)*(eta_max-eta_min)/n_s)*(2*i*n_s<=j & j <(2*i+1)*n_s);
     term_1 = term1+term_1;
     term2 = (eta_max -( j-(2*i+1)*n_s)*(eta_max-eta_min)/n_s)*( (2*i+1)*n_s<=j & j <2*(i+1)*n_s);  
     term_2=term_2+term2;
    end 
 eta_t(j)=term_1+term_2;   
end 
 
end


%%%%%%%%%%%%%%%%%%
%% With Batch Normalization
%%%%%%%%%%%%%%%%%%
addpath Datasets\cifar-10-batches-mat;
clear
clc


%% Load Data
[train_X1,train_Y1,train_y1]=LoadBatch('data_batch_1.mat');
[train_X2,train_Y2,train_y2]=LoadBatch('data_batch_2.mat');
[train_X3,train_Y3,train_y3]=LoadBatch('data_batch_3.mat');
[train_X4,train_Y4,train_y4]=LoadBatch('data_batch_4.mat');
[train_X5,train_Y5,train_y5]=LoadBatch('data_batch_5.mat');
% 
train_X=[train_X1,train_X2(:,5001:size(train_X2,2)),train_X3, train_X4, train_X5];
train_Y=[train_Y1,train_Y2(:,5001:size(train_X2,2)),train_Y3, train_Y4, train_Y5];
train_y=[train_y1,train_y2(:,5001:size(train_X2,2)),train_y3, train_y4, train_y5];

validation_X=train_X2(:,1:5000);
validation_Y=train_Y2(:,1:5000);
validation_y=train_y2(:,1:5000);

[testX,testY,testy]=LoadBatch('test_batch.mat');

%% 
% [train_X,train_Y,train_y]=LoadBatch('data_batch_1.mat');
% [validation_X,validation_Y,validation_y]=LoadBatch('data_batch_2.mat');
% [testX,testY,testy]=LoadBatch('test_batch.mat');

%%
mean_X = mean (train_X,  2);
std_X = std(train_X, 0, 2);
train_X = train_X - repmat(mean_X,[1, size(train_X,2)]);
train_X = train_X./ repmat(std_X,[1, size(train_X,2)]);

validation_X = validation_X - repmat(mean_X,[1, size(validation_X,2)]);
validation_X = validation_X./ repmat(std_X,[1, size(validation_X,2)]);

testX = testX - repmat(mean_X,[1, size(testX,2)]);
testX = testX./ repmat(std_X,[1, size(testX,2)]);

%% Initialize Parameters
m = [50,50];
batch_size = 100;
dimension = 10;
k = size(m,2)+1;
% n_searching = 50;
% for i = 1:n_searching
%  l_min = -3;
%  l_max = -1.4;
%  l = l_min + (l_max - l_min) * rand(1,1);
%  lambda = 10^l;
% end
lambda =0.005;
delta = 1e-6;


%% Check gradients
% gradCheck(dimension,batch_size, train_X, train_Y, lambda, delta, k,m)

%% Calculate Accuracy
n_batch = 100;
eta_min = 1e-5;
eta_max = 1e-1;
n_s = 5* 45000/n_batch;
t =4 *n_s;
N =size(train_X,2);
eta_t =  LearningRate(eta_min, eta_max, n_s,t);
% N = 100;
J_validation = zeros(1,t/N*n_batch);
J_train = zeros(1,t/N*n_batch);
accuracy_train = zeros(1,t/N*n_batch);
accuracy_validation = zeros(1,t/N*n_batch);
[W, b] = Initializing_network(train_X,m);
[gamma,beta] = Initializing_gamma_beta(train_X,W,b,k);
count = 0;


for e=1:t/N*n_batch
    tic;
        J_train(e) = ComputeCost(train_X, train_Y, W, b, gamma,beta,lambda,k);
        J_validation(e) = ComputeCost(validation_X, validation_Y, W, b,gamma,beta, lambda,k);
        
        accuracy_train(e) = ComputeAccuracy(train_X,train_y,W,b,gamma,beta,k);
        accuracy_validation(e) = ComputeAccuracy(validation_X,validation_y,W,b,gamma,beta,k);
        
        
   
     
        [Wstar,bstar,gammastar,betastar] = MiniBatchGD(train_X,train_Y, W,b,gamma,beta, N,k,count,n_batch, eta_t,lambda);
        W=Wstar;
        b=bstar;
        gamma = gammastar;
        beta = betastar;
        count = count+1;
        

        
        % shuffle data
        index = randperm(size(train_X,2));
        train_X = train_X(:,index);
        train_Y = train_Y(:,index);
        train_y = train_y(:,index);
        toc

end 



figure()
plot (1:t/N*batch_size, J_train, 'g')
hold on
plot (1:t/N*batch_size, J_validation, 'r')
hold off
legend('training cost','validation cost')
legend('training cost')
xlabel('epochs')
ylabel('cost')



figure()
plot (1:t/N*batch_size, accuracy_train, 'b')
hold on
plot (1:t/N*batch_size, accuracy_validation, 'r')
hold off
legend('training accuracy','validation accuracy')
xlabel('epochs')
ylabel('accuracy')


disp(['lambda:' num2str(lambda)])
accuracy_train = ComputeAccuracy(train_X,train_y,W,b,gamma,beta,k);
disp(['Training Accuracy:' num2str(accuracy_train*100) '%'])

accuracy_test = ComputeAccuracy(testX,testy,W,b,gamma,beta,k);
disp(['Test Accuracy:' num2str(accuracy_test*100) '%'])




%%
function gradCheck(dimension,batch_size, X, Y, lambda, delta, k,m)
[W, b] = Initializing_network(X(1:dimension, 1 : batch_size),m);
[gamma,beta] = Initializing_gamma_beta(X(1:dimension, 1 : batch_size),W,b,k);
% numerical gradients
[~,~,~,mu,v] = intervalues(X(1:dimension, 1 : batch_size), W, b,gamma,beta, k);
NetParams.use_bn=0;
NetParams.W = W;
NetParams.b = b;
NetParams.gammas = gamma;
NetParams.betas = beta;
[ngrad_W,ngrad_b] = ComputeGradsNumSlow(X(1:dimension, 1 : batch_size), ...
    Y(1:dimension, 1 : batch_size), NetParams, lambda, delta,k);

% analytical gradients
[h,s,scap,mu,v] = intervalues(X(1:dimension, 1 : batch_size), W, b,gamma,beta,k);
P = EvaluateClassifier(h, W, b);
[grad_W, grad_b,grad_gamma,grad_beta] = ComputeGradients(X(1:dimension, 1 : batch_size), ...
    Y(1:dimension, 1 : batch_size),  W, b,gamma,beta, mu,v,k, lambda);

% relative error rate
eps = 1e-10;
for i = 1 : length(W)
    gradcheck_bi = sum(abs(ngrad_b{i} - grad_b{i})/max(eps, sum(abs(ngrad_b{i}) + abs(grad_b{i}))));
    gradcheck_bm = max(abs(ngrad_b{i} - grad_b{i})/max(eps, sum(abs(ngrad_b{i}) + abs(grad_b{i}))));
    gradcheck_Wi = sum(sum(abs(ngrad_W{i} - grad_W{i})/max(eps, sum(sum(abs(ngrad_W{i}) + abs(grad_W{i}))))));
    gradcheck_Wm = max(max(abs(ngrad_W{i} - grad_W{i})/max(eps, sum(sum(abs(ngrad_W{i}) + abs(grad_W{i}))))));
    disp(['error of grad_W' num2str(i) ': ' num2str(gradcheck_Wi)])
    disp(['max error of grad_W' num2str(i) ': ' num2str(gradcheck_Wm)])
    disp(['error of grad_b' num2str(i) ': ' num2str(gradcheck_bi)])
    disp(['max error of grad_b' num2str(i) ': ' num2str(gradcheck_bm)])
end
for i = 1 : length(gamma)
    gradcheck_betai = sum(abs(ngrad_beta{i} - grad_beta{i})/max(eps, sum(abs(ngrad_beta{i}) + abs(grad_beta{i}))));
    gradcheck_bm = max(abs(ngrad_beta{i} - grad_beta{i})/max(eps, sum(abs(ngrad_beta{i}) + abs(grad_beta{i}))));
    gradcheck_gammai = sum(sum(abs(ngrad_gamma{i} - grad_gamma{i})/max(eps, sum(sum(abs(ngrad_gamma{i}) + abs(grad_gamma{i}))))));
    gradcheck_Wm = max(max(abs(ngrad_gamma{i} - grad_gamma{i})/max(eps, sum(sum(abs(ngrad_gamma{i}) + abs(grad_gamma{i}))))));
    disp(['error of grad_W' num2str(i) ': ' num2str(gradcheck_gammai)])
    disp(['max error of grad_W' num2str(i) ': ' num2str(gradcheck_gammam)])
    disp(['error of grad_b' num2str(i) ': ' num2str(gradcheck_betai)])
    disp(['max error of grad_b' num2str(i) ': ' num2str(gradcheck_betam)])
end
end


%% load the data
function [X, Y, y] = LoadBatch(filename)
Dataset = load(filename);
data = Dataset.data';
label = Dataset.labels;
newlabel = label + 1;

Y_array = zeros(10,10000);

for i = 1:10000
    Y_array(newlabel(i),i)=1;
end

X = double(data)/255;
Y = double(Y_array);
y = double(newlabel');
end

% initialize the parameters
%%
function [W,b] = Initializing_network(X,m)
d = size(X,1);
% sig=1e-4;
% W{1} = sig.*randn(m(1), d);
W{1} = 1/sqrt(d).*randn(m(1), d);
b{1} = zeros(m(1),1);
for i = 1:size(m,2)
%     std = sig;
     %Xavier initialization
    std = sqrt(1/m(i));
 if i~=size(m,2)
    W{i+1} = std.* randn (m(i+1),m(i));
    b{i+1} = zeros(m(i+1),1);
 else    W{i+1} = std.*randn(10,m(end));
    b{i+1} = zeros(10,1);
 end
end
end


%% Intervalues
function [h,s,scap,mu,v]=intervalues(X,W,b,gamma,beta,k)
eps=0.000001;
for l = 1:k-1
 s{l} = W{l}*X+b{l};
[scap{l}, mu{l}, v{l}] = BN_forward(s{l}, eps);

sbar{l} = gamma{l}.*scap{l}+repmat(beta{l},1,size(s{l},2));
X = max(0,sbar{l});
h{l} = X;
end
end 


function [gamma,beta] = Initializing_gamma_beta(X,W,b,k)
eps=0.000001;
for l = 1:k-1
 s{l} = W{l}*X+b{l};
[scap{l}, mu{l}, v{l}] = BN_forward(s{l}, eps);
gamma{l} = (v{l} + eps).^(0.5);
beta{l} =mu{l};
sbar{l} = gamma{l}.*scap{l}+repmat(beta{l},1,size(s{l},2));
X = max(0,sbar{l});
h{l} = X;
end

end
%% BN_forward
function [scap, mu, v] = BN_forward(s, eps, mu_av, v_av)
if nargin < 4
    mu = mean(s, 2);
    v = mean((s - repmat(mu, 1, size(s, 2))).^2, 2);
else
    mu = mu_av;
    v = v_av;
end

scap = diag((v + eps).^(-0.5))*(s - repmat(mu, 1, size(s, 2)));

end


%%
function P = EvaluateClassifier(h, W, b)

W = W{end};
b = b{end};
X = h{end};
b = repmat(b, 1, size(X, 2));

s = W*X + b;

P  = exp(s)./sum(exp(s));


end


%% ComputeCost
function J = ComputeCost(X, Y, W, b,gamma,beta, lambda, k)

[h,~,~,~,~]=intervalues(X,W,b,gamma,beta,k);

P = EvaluateClassifier(h, W, b);

    sumP = 0;
    J2=0;
    for i = 1:size(Y,2)
        sumP =   sumP + (-log(Y(:,i)'*P(:,i)));
    end
    J1 = 1/size(X,2)*sumP ;
    
for i = 1 : length(W)
    temp = W{i}.^2;
    J2 = J2 + lambda*sum(temp(:));
end

    J = J1+ J2;
end


%% ComputeAccuracy
function acc = ComputeAccuracy(X, y, W, b,gamma,beta,k)

[h,~,~,~,~]=intervalues(X,W,b,gamma,beta,k);
P = EvaluateClassifier(h, W, b);
[~, ind] = max(P);
acc = length(find(y - ind == 0))/length(y);
end

%% BN_backward
       
function G = BN_backward(G, s, mu, v,  eps)
sigma1 = (v+eps).^(-0.5);
sigma2 = (v+eps).^(-1.5);

G1 = G.*(sigma1*ones(size(s,2),1)');
G2 = G.*(sigma2*ones(size(s,2),1)');
D = s - mu*ones(size(s,2),1)';
c=(G2.*D)*ones(size(s,2),1);
G = G1- G1*ones(size(s,2),1)/size(s,2)-D/size(s,2).*(c*ones(size(s,2),1)');

end


    
%% ComputeGradients
function [grad_W, grad_b,grad_gamma,grad_beta] = ComputeGradients(X, Y, W,b,gamma,beta,P,h,s,scap,mu,v,k,lambda)

eps = 0.001;
G = -(Y-P);
grad_W{k}= G*h{k-1}'/size(X,2)+2* lambda*W{k};
grad_b{k} = G* ones(size(X,2),1)/size(X,2);
 G = W{k}'* G;
 Ind = h{k -1}>0;
 G = G.* Ind;
 for l= k-1:-1:1
     grad_gamma{l} =( G.*scap{l})*ones(size(X,2),1)/size(X,2);
     grad_beta{l} = G* ones(size(X,2),1)/size(X,2);
     G = G.*(gamma{l}*ones(size(X,2),1)');
     G = BN_backward(G, s{l},mu{l}, v{l},  eps);
     grad_b{l} = G* ones(size(X,2),1)/size(X,2);
     if l ==1
     grad_W{l}= G*X'/size(X,2)+2* lambda*W{l};  
     else
     grad_W{l}= G*h{l-1}'/size(X,2)+2* lambda*W{l};
     end
    if l > 1
     G = W{l}' *G;
     Ind = h{l -1}>0;
     G = G.* Ind;    
     end
 end

end


%% [Wstar, bstar] = MiniBatchGD(trainX, trainY, GDparams, W, b, lambda);
 function [Wstar,bstar,gammastar,betastar] = MiniBatchGD(X,Y, W,b,gamma,beta, N,k,count,n_batch, eta_t,lambda)
 

alpha =0.9;


for j=1:N/n_batch
        j_start = (j-1)*n_batch + 1;
        j_end = j*n_batch;
        inds = j_start:j_end;
        Xbatch = X(:, inds);
        Ybatch = Y(:, inds);
        index = j + count* N/n_batch;
      
        [h,s,scap,mu_out,v_out]=intervalues(Xbatch,W,b,gamma,beta,k);
        P = EvaluateClassifier(h, W, b);
    for i = 1 : k - 1
        if j == 1
            mu = mu_out;
            v = v_out;
        else
            mu{i} = alpha*mu{i} + (1 - alpha)*mu_out{i};
            v{i} = alpha*v{i} + (1 - alpha)*v_out{i};
        end
    end

        [grad_W, grad_b,grad_gamma,grad_beta] = ComputeGradients(Xbatch, Ybatch,W, b,gamma,beta,P,h,s,scap,mu,v,k,lambda);
        for l = 1:k-1
        W{l}= W{l} - eta_t(index) .* grad_W{l};
        b{l} = b{l} - eta_t (index).* grad_b{l}; 
        gamma{l}= gamma{l} - eta_t(index) .* grad_gamma{l};
        beta{l} = beta{l} - eta_t (index).* grad_beta{l};     
        end 
        W{k}= W{k} - eta_t(index) .* grad_W{k};
        b{k} = b{k} - eta_t (index).* grad_b{k};        
end
 
Wstar = W;
bstar = b;
gammastar = gamma;
betastar = beta;

 end 


%% ComputeGradsNumSlow
function [grad_W,grad_b] = ComputeGradsNumSlow(X, Y, NetParams, lambda, h,k)

grad_W = cell(numel(NetParams.W), 1);
grad_b = cell(numel(NetParams.b), 1);
if NetParams.use_bn
    grad_gammas = cell(numel(NetParams.gammas), 1);
    grad_betas = cell(numel(NetParams.betas), 1);
end

for j=1:length(NetParams.b)
    grad_b{j} = zeros(size(NetParams.b{j}));
    NetTry = NetParams;
    for i=1:length(NetParams.b{j})
        b_try = NetParams.b;
        b_try{j}(i) = b_try{j}(i) - h;
        NetTry.b = b_try;
        c1 = ComputeCost(X, Y, NetParams.W,NetTry.b, lambda,k);        
        
        b_try = NetParams.b;
        b_try{j}(i) = b_try{j}(i) + h;
        NetTry.b = b_try;        
        c2 = ComputeCost(X, Y, NetParams.W,NetTry.b, lambda,k);
        
        grad_b{j}(i) = (c2-c1) / (2*h);
    end
end

for j=1:length(NetParams.W)
    grad_W{j} = zeros(size(NetParams.W{j}));
        NetTry = NetParams;
    for i=1:numel(NetParams.W{j})
        
        W_try = NetParams.W;
        W_try{j}(i) = W_try{j}(i) - h;
        NetTry.W = W_try;        
        c1 = ComputeCost(X, Y, NetTry.W,NetParams.b, lambda,k);
    
        W_try = NetParams.W;
        W_try{j}(i) = W_try{j}(i) + h;
        NetTry.W = W_try;        
        c2 = ComputeCost(X, Y, NetTry.W,NetParams.b, lambda,k);
    
        grad_W{j}(i) = (c2-c1) / (2*h);
    end
end

if NetParams.use_bn
    for j=1:length(NetParams.gammas)
        grad_gammas{j} = zeros(size(NetParams.gammas{j}));
        NetTry = NetParams;
        for i=1:numel(NetParams.gammas{j})
            
            gammas_try = NetParams.gammas;
            gammas_try{j}(i) = gammas_try{j}(i) - h;
            NetTry.gammas = gammas_try;        
            c1 = ComputeCost(X, Y, NetTry.gammas,NetParams.betas, lambda,k);
            
            gammas_try = NetParams.gammas;
            gammas_try{j}(i) = gammas_try{j}(i) + h;
            NetTry.gammas = gammas_try;        
            c2 = ComputeCost(X, Y, NetTry.gammas, NetParams.betas,lambda,k);
            
            grad_gammas{j}(i) = (c2-c1) / (2*h);
        end
    end
    
    for j=1:length(NetParams.betas)
        grad_betas{j} = zeros(size(NetParams.betas{j}));
        NetTry = NetParams;
        for i=1:numel(NetParams.betas{j})
            
            betas_try = NetParams.betas;
            betas_try{j}(i) = betas_try{j}(i) - h;
            NetTry.betas = betas_try;        
            c1 = ComputeCost(X, Y,NetParams.gammas, NetTry.betas, lambda,k);
            
            betas_try = NetParams.betas;
            betas_try{j}(i) = betas_try{j}(i) + h;
            NetTry.betas = betas_try;        
            c2 = ComputeCost(X, Y, NetParams.gammas,NetTry.betas, lambda,k);
            
            grad_betas{j}(i) = (c2-c1) / (2*h);
        end
    end    
end 
end

%% Learning rate
function eta_t = LearningRate(eta_min, eta_max, n_s,t)
eta_t= zeros(1,t);

for j = 1: t
    term_1 = 0;
    term_2 = 0;
    for i= 0:5
     term1 = (eta_min +( j-2*i*n_s)*(eta_max-eta_min)/n_s)*(2*i*n_s<=j & j <(2*i+1)*n_s);
     term_1 = term1+term_1;
     term2 = (eta_max -( j-(2*i+1)*n_s)*(eta_max-eta_min)/n_s)*( (2*i+1)*n_s<=j & j <2*(i+1)*n_s);  
     term_2=term_2+term2;
    end 
 eta_t(j)=term_1+term_2;   
end 
 
end

