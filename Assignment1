
addpath Datasets\cifar-10-batches-mat;

% P2
sigma=0.01;
mean = 0;
K=10;
d=3072;
W=sigma.*randn(K,d) + mean;
b=sigma.*randn(K,1) + mean;
eps=1e-10;
[trainX,trainY,trainy]=LoadBatch('data_batch_1.mat');
[validationX,validationY,validationy]=LoadBatch('data_batch_2.mat');
[testX,testY,testy]=LoadBatch('test_batch.mat');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
P=EvaluateClassifier(trainX(:,1:20),W,b);
[ngrad_W, ngrad_b]=ComputeGradsNumSlow(trainX(:,1:20),trainY(:,1:20),W,b,0,1e-6);
[grad_W,grad_b]=ComputeGradients(trainX(:,1:20),trainY(:,1:20),P,W,0);

Error_W=max(max(abs(grad_W-ngrad_W)./max(eps,(abs(grad_W)+abs(ngrad_W)))));
Error_b=max(max(abs(grad_b-ngrad_b)./max(eps,(abs(grad_b)+abs(ngrad_b)))));

if Error_W<1e-3
    fprintf("W has a correct gradient!")
else
    fprintf("W has a wrong gradient!")
end

if Error_b<1e-6
    fprintf("b has a correct gradient!")
else
    fprintf("b has a wrong gradient!")
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

lambda = 0;
GDparams.eta = 0.1;
GDparams.n_batch = 100;
GDparams.n_epochs = 40;
% decay = 0.9;

J_validation = zeros(1, GDparams.n_epochs);
J_train = zeros(1, GDparams.n_epochs);
for j = 1: GDparams.n_epochs
    J_train(j) = ComputeCost(trainX, trainY, W, b, lambda);
    J_validation(j) = ComputeCost(validationX, validationY, W, b, lambda);
    [Wstar, bstar] = MiniBatchGD(trainX, trainY, GDparams, W, b, lambda);
    W=Wstar;
    b=bstar;
    % GDparams.eta = decay * GDparams.eta;
end
figure()
plot (1:GDparams.n_epochs, J_train, 'g')
hold on
plot (1:GDparams.n_epochs, J_validation, 'r')
hold off
legend('training loss','validation loss')
xlabel('epochs')
ylabel('loss')

accuracy_train = ComputeAccuracy(trainX,trainy,W,b);
disp(['Training Accuracy:' num2str(accuracy_train) '%'])

accuracy_test = ComputeAccuracy(testX,testy,W,b);
disp(['Test Accuracy:' num2str(accuracy_test) '%'])

K=size(W,1);
for i=1:K
    im = reshape(W(i,:),32,32,3);
    s_im{i} = (im - min(im(:))) / (max(im(:)) - min (im(:)));
    s_im{i} = permute(s_im{i}, [2,1,3]);
end
figure()
montage(s_im,'size',[1,K])
% P1
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


% P3
function P=EvaluateClassifier(X,W,b)
W = double(W);
X = double(X);
b = double(b);
s=W*X+b;
P=bsxfun(@rdivide, exp(s), sum(exp(s), 1));
end


% P4
function J=ComputeCost(X,Y,W,b,lambda)
[~,N]=size(X);
P=EvaluateClassifier(X,W,b);
crossentropy=0;
for i=1:N
    y=Y(:,i);
    p=P(:,i);
    crossentropy=crossentropy-log(y.'*p);
end
J=1./N*crossentropy+lambda*sumsqr(W);
end

% P5

function acc = ComputeAccuracy(X, y, W, b)

P = EvaluateClassifier(X, W, b);
[~, ind] = max(P);
acc = length(find(y - ind == 0))/length(y)*100;
end

% P6
function[grad_W,grad_b]=ComputeGradients(X,Y,P,W,lambda)
grad_W=zeros(size(W));
grad_b=zeros(size(W,1),1);
t=size(X,2);
for i= 1: t
    x=X(:,i);
    y=Y(:,i);
    p=P(:,i);
    g = -y'/(y'*p)*(diag(p)-p*p');
    grad_b = grad_b + g';  
    grad_W = grad_W + g'* x';
end
grad_b=grad_b/t;
grad_W=grad_W/t+2*lambda*W;
end

% ComputeGradsNumSlow
function [grad_W, grad_b] = ComputeGradsNumSlow(X, Y, W, b, lambda, h)

no = size(W, 1);
d = size(X, 1);

grad_W = zeros(size(W));
grad_b = zeros(no, 1);

for i=1:length(b)
    b_try = b;
    b_try(i) = b_try(i) - h;
    c1 = ComputeCost(X, Y, W, b_try, lambda);
    b_try = b;
    b_try(i) = b_try(i) + h;
    c2 = ComputeCost(X, Y, W, b_try, lambda);
    grad_b(i) = (c2-c1) / (2*h);
end

for i=1:numel(W)
    
    W_try = W;
    W_try(i) = W_try(i) - h;
    c1 = ComputeCost(X, Y, W_try, b, lambda);
    
    W_try = W;
    W_try(i) = W_try(i) + h;
    c2 = ComputeCost(X, Y, W_try, b, lambda);
    
    grad_W(i) = (c2-c1) / (2*h);
end
end

% P7
function [Wstar,bstar] = MiniBatchGD(X,Y,GDparams,W,b,lambda)
eta = GDparams.eta;
n_batch = GDparams.n_batch;
n_epoch = GDparams.n_epochs;
N = size(X,2);
for j=1:N/n_batch
        j_start = (j-1)*n_batch + 1;
        j_end = j*n_batch;
        inds = j_start:j_end;
        Xbatch = X(:, inds);
        Ybatch = Y(:, inds);
        P = EvaluateClassifier(Xbatch, W, b);
        [grad_W, grad_b] = ComputeGradients(Xbatch, Ybatch, P, W, lambda);
        W = W - eta * grad_W;
        b = b - eta * grad_b;
end 
Wstar = W;
bstar = b;
end


addpath Datasets\cifar-10-batches-mat;


% P2
sigma=0.01;
mean = 0;
K=10;
d=3072;
W=sigma.*randn(K,d) + mean;
b=sigma.*randn(K,1) + mean;
eps=1e-10;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Use all the available training data for training (all five batches minus a small
% subset of the training images for a validation set). Decrease the size of the
% validation set down to 1000.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[train_X1,train_Y1,train_y1]=LoadBatch('data_batch_1.mat');
[train_X2,train_Y2,train_y2]=LoadBatch('data_batch_2.mat');
[train_X3,train_Y3,train_y3]=LoadBatch('data_batch_3.mat');
[train_X4,train_Y4,train_y4]=LoadBatch('data_batch_4.mat');
[train_X5,train_Y5,train_y5]=LoadBatch('data_batch_5.mat');

N=size(train_X2,2);
train_X=[train_X1,train_X2(:,1001:N),train_X3,train_X4,train_X5];
train_Y=[train_Y1,train_Y2(:,1001:N),train_Y3,train_Y4,train_Y5];
train_y=[train_y1,train_y2(:,1001:N),train_y3,train_y4,train_y5];

validation_X=train_X2(:,1:1000);
validation_Y=train_Y2(:,1:1000);
validation_y=train_y2(:,1:1000);


[testX,testY,testy]=LoadBatch('test_batch.mat');

lambda = 0.1;
GDparams.eta = 0.01;
GDparams.n_batch = 100;
GDparams.n_epochs = 40;
decay = 0.9;



J_validation = zeros(1, GDparams.n_epochs);
J_train = zeros(1, GDparams.n_epochs);
for j = 1: GDparams.n_epochs
    J_train(j) = ComputeCost(train_X, train_Y, W, b, lambda);
    J_validation(j) = ComputeCost(validation_X, validation_Y, W, b, lambda);
    [Wstar, bstar] = MiniBatchGD(train_X, train_Y, GDparams, W, b, lambda);
    W=Wstar;
    b=bstar;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Play around with decaying the learning rate by a factor 0.9 after each epoch.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    GDparams.eta = decay * GDparams.eta; % decay with 0.9;
    
   
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Shuffle the order of your training examples at the beginning of every epoch.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    index = randperm(size(train_X,2));
    train_X = train_X(:,index);
    train_Y = train_Y(:,index);
    train_y = train_y(:,index);
end
figure()
plot (1:GDparams.n_epochs, J_train, 'g');
hold on
plot (1:GDparams.n_epochs, J_validation, 'r');
hold off
legend('training loss','validation loss')
xlabel('epochs')
ylabel('loss')

accuracy_train = ComputeAccuracy(train_X,train_y,W,b);
disp(['Training Accuracy:' num2str(accuracy_train) '%'])

accuracy_test = ComputeAccuracy(testX,testy,W,b);
disp(['Test Accuracy:' num2str(accuracy_test) '%'])

K=size(W,1);
for i=1:K
    im = reshape(W(i,:),32,32,3);
    s_im{i} = (im - min(im(:))) / (max(im(:)) - min (im(:)));
    s_im{i} = permute(s_im{i}, [2,1,3]);
end
figure();
montage(s_im,'size',[1,K])
% P1

function [X, Y, y] = LoadBatch(filename)

dataset = load(filename);
data = dataset.data';
label = dataset.labels;
newlabel = label + 1;

 
Y_array = zeros(10,10000);

for i = 1:10000
    Y_array(newlabel(i),i)=1;
end

X = double(data)/255;
Y = double(Y_array);
y = double(newlabel');
end


% P3
function P=EvaluateClassifier(X,W,b)
W = double(W);
X = double(X);
b = double(b);
s=W*X+b;
P=bsxfun(@rdivide, exp(s), sum(exp(s), 1));
end


% P4
function J=ComputeCost(X,Y,W,b,lambda)
[~,N]=size(X);
P=EvaluateClassifier(X,W,b);
crossentropy=0;
for i=1:N
    y=Y(:,i);
    p=P(:,i);
    crossentropy=crossentropy-log(y.'*p);
end
J=1./N*crossentropy+lambda*sumsqr(W);
end

% P5

function acc = ComputeAccuracy(X, y, W, b)

P = EvaluateClassifier(X, W, b);
[~, ind] = max(P);
acc = length(find(y - ind == 0))/length(y)*100;
end

% P6
function[grad_W,grad_b]=ComputeGradients(X,Y,P,W,lambda)
grad_W=zeros(size(W));
grad_b=zeros(size(W,1),1);
t=size(X,2);
for i= 1: t
    x=X(:,i);
    y=Y(:,i);
    p=P(:,i);
    g = -y'/(y'*p)*(diag(p)-p*p');
    grad_b = grad_b + g';  
    grad_W = grad_W + g'* x';
end
grad_b=grad_b/t;
grad_W=grad_W/t+2*lambda*W;
end


% function [grad_W, grad_b] = ComputeGradsNumSlow(X, Y, W, b, lambda, h)
% 
% no = size(W, 1);
% d = size(X, 1);
% 
% grad_W = zeros(size(W));
% grad_b = zeros(no, 1);
% 
% for i=1:length(b)
%     b_try = b;
%     b_try(i) = b_try(i) - h;
%     c1 = ComputeCost(X, Y, W, b_try, lambda);
%     b_try = b;
%     b_try(i) = b_try(i) + h;
%     c2 = ComputeCost(X, Y, W, b_try, lambda);
%     grad_b(i) = (c2-c1) / (2*h);
% end
% 
% for i=1:numel(W)
%     
%     W_try = W;
%     W_try(i) = W_try(i) - h;
%     c1 = ComputeCost(X, Y, W_try, b, lambda);
%     
%     W_try = W;
%     W_try(i) = W_try(i) + h;
%     c2 = ComputeCost(X, Y, W_try, b, lambda);
%     
%     grad_W(i) = (c2-c1) / (2*h);
% end
% end

% P 7
function [Wstar,bstar] = MiniBatchGD(X,Y,GDparams,W,b,lambda)
eta = GDparams.eta;
n_batch = GDparams.n_batch;
n_epoch = GDparams.n_epochs;
N = size(X,2);
for j=1:N/n_batch
        j_start = (j-1)*n_batch + 1;
        j_end = j*n_batch;
        inds = j_start:j_end;
        Xbatch = X(:, inds);
        Ybatch = Y(:, inds);
        P = EvaluateClassifier(Xbatch, W, b);
        [grad_W, grad_b] = ComputeGradients(Xbatch, Ybatch, P, W, lambda);
        W = W - eta * grad_W;
        b = b - eta * grad_b;
end 
Wstar = W;
bstar = b;
end

addpath Datasets\cifar-10-batches-mat;

% P2
sigma=0.01;
mean = 0;
K=10;
d=3072;
W=sigma.*randn(K,d) + mean;
b=sigma.*randn(K,1) + mean;
eps=1e-10;
[trainX,trainY,trainy]=LoadBatch('data_batch_1.mat');
[validationX,validationY,validationy]=LoadBatch('data_batch_2.mat');
[testX,testY,testy]=LoadBatch('test_batch.mat');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% s=EvaluateClassifier(trainX(:,1:20),W);
% ngrad_W=ComputeGradsNumSlow(trainX(:,1:20),trainY(:,1:20),W,0,1,1e-6);
% grad_W=ComputeGradients(trainX(:,1:20),trainY(:,1:20),s,W,1,0);
% 
% Error_W=max(max(abs(grad_W-ngrad_W)./max(eps,(abs(grad_W)+abs(ngrad_W)))));;
% 
% if Error_W<1.2
%     fprintf("W has a correct gradient!")
% else
%     fprintf("W has a wrong gradient!")
% end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

lambda = 0.1;
GDparams.eta = 0.01;
GDparams.n_batch = 40;
GDparams.n_epochs = 100;
delta = 1;
decay = 0.9;

J_validation = zeros(1, GDparams.n_epochs);
J_train = zeros(1, GDparams.n_epochs);

for j = 1: GDparams.n_epochs
    J_train(j) = ComputeCost(trainX, trainY, W, lambda, delta);
    J_validation(j) = ComputeCost(validationX, validationY, W,lambda, delta);
    Wstar = MiniBatchGD(trainX, trainY, GDparams, W, delta, lambda);
    W=Wstar;
    
    GDparams.eta = decay * GDparams.eta;
end
figure()
plot (1:GDparams.n_epochs, J_train, 'g')
hold on
plot (1:GDparams.n_epochs, J_validation, 'r')
hold off
legend('training loss','validation loss')
xlabel('epochs')
ylabel('loss')

accuracy_train = ComputeAccuracy(trainX,trainy,W);
disp(['Training Accuracy:' num2str(accuracy_train) '%'])

accuracy_test = ComputeAccuracy(testX,testy,W);
disp(['Test Accuracy:' num2str(accuracy_test) '%'])

K=size(W,1);
for i=1:K
    im = reshape(W(i,:),32,32,3);
    s_im{i} = (im - min(im(:))) / (max(im(:)) - min (im(:)));
    s_im{i} = permute(s_im{i}, [2,1,3]);
end
figure()
montage(s_im,'size',[1,K])

% P1
function [X, Y, y] = LoadBatch(filename)
dataset = load(filename);
data = dataset.data';
label = dataset.labels;
newlabel = label + 1;

Y_array = zeros(10,10000);

for i = 1:10000
    Y_array(newlabel(i),i)=1;
end

X = double(data)/255;
Y = double(Y_array);
y = double(newlabel');
end


% P3
% s->10*10000
function s=EvaluateClassifier(X,W)
W = double(W);
X = double(X);
s=W*X;
end


% P4

function J = ComputeCost(X, Y, W, lambda, delta)

s = EvaluateClassifier(X, W);
% s->10*10000; Y->10*10000; s_sum->1*10000;sc->10*10000
s_sum = sum(s.*Y)
s_y = repmat(s_sum, size(s, 1), 1);
margin = s - s_y + delta;
J_W = sum(margin(find(margin > 0))) - size(s, 2)*delta;
J_W = J_W/size(s, 2);
J_regular = lambda*sum(sum(W.^2));
J = J_W + J_regular;

end

% P5
function acc = ComputeAccuracy(X, y, W)

s = EvaluateClassifier(X, W);
[~, ind] = max(s);
acc = length(find(y - ind == 0))/length(y)*100;
end

% P6

function grad_W = ComputeGradients(X, Y, s, W, delta, lambda)
grad_W = zeros(size(W));
s_sum=sum(s.*Y);
s_y = repmat(s_sum, size(s, 1), 1);
margin = s - s_y + delta;
flag = zeros(size(s));
flag(find(margin > 0)) = 1;
flag(find(Y == 1)) = -1;
N= size(X,2);
for i = 1 : N
    x = X(:, i);
    f = flag(:, i);
    g = repmat(x', size(W, 1), 1);
    g(find(f == 0), :) = 0;
    g(find(f == -1), :) = -length(find(f == 1))*g(find(f == -1), :);
    grad_W = grad_W + g;
end

grad_W = 2*lambda*W + grad_W/N;

end

% ComputeGradsNumSlow
function grad_W = ComputeGradsNumSlow(X, Y, W, lambda,delta,h)

no = size(W, 1);
d = size(X, 1);

grad_W = zeros(size(W));

for i=1:numel(W)
    
    W_try = W;
    W_try(i) = W_try(i) - h;
    c1 = ComputeCost(X, Y, W_try, lambda,delta);
    
    W_try = W;
    W_try(i) = W_try(i) + h;
    c2 = ComputeCost(X, Y, W_try,lambda,delta);
    
    grad_W(i) = (c2-c1) / (2*h);
end
end

% P 7
function Wstar = MiniBatchGD(X,Y,GDparams,W,delta,lambda)
eta = GDparams.eta;
n_batch = GDparams.n_batch;
n_epoch = GDparams.n_epochs;
N = size(X,2);
for j=1:N/n_batch
        j_start = (j-1)*n_batch + 1;
        j_end = j*n_batch;
        inds = j_start:j_end;
        Xbatch = X(:, inds);
        Ybatch = Y(:, inds);
        s = EvaluateClassifier(Xbatch, W);
        grad_W = ComputeGradients(Xbatch, Ybatch, s, W, delta, lambda);
        W = W - eta * grad_W;
        
end 
Wstar = W;

end
