addpath Datasets\cifar-10-batches-mat;



K=10;
d=3072;
m= 50;
[W, b] = Initializing_network(d, m, K);


eps=1e-10;




[train_X1,train_Y1,train_y1]=LoadBatch('data_batch_1.mat');
[train_X2,train_Y2,train_y2]=LoadBatch('data_batch_2.mat');
[train_X3,train_Y3,train_y3]=LoadBatch('data_batch_3.mat');
[train_X4,train_Y4,train_y4]=LoadBatch('data_batch_4.mat');
[train_X5,train_Y5,train_y5]=LoadBatch('data_batch_5.mat');

train_X=[train_X1,train_X2(:,5001:size(train_X2,2)),train_X3];
train_Y=[train_Y1,train_Y2(:,5001:size(train_X2,2)),train_Y3];
train_y=[train_y1,train_y2(:,5001:size(train_X2,2)),train_y3];

validation_X=train_X2(:,1:5000);
validation_Y=train_Y2(:,1:5000);
validation_y=train_y2(:,1:5000);

[testX,testY,testy]=LoadBatch('test_batch.mat');

% [train_X,train_Y,train_y]=LoadBatch('data_batch_1.mat');
% [validation_X,validation_Y,validation_y]=LoadBatch('data_batch_2.mat');
% [testX,testY,testy]=LoadBatch('test_batch.mat');
% 
% mean_X = mean (train_X,  2);
% std_X = std(train_X, 0, 2);
% train_X = train_X - repmat(mean_X,[1, size(train_X,2)]);
% train_X = train_X./ repmat(std_X,[1, size(train_X,2)]);
% 
% validation_X = validation_X - repmat(mean_X,[1, size(validation_X,2)]);
% validation_X = validation_X./ repmat(std_X,[1, size(validation_X,2)]);
% 
% testX = testX - repmat(mean_X,[1, size(testX,2)]);
% testX = testX./ repmat(std_X,[1, size(testX,2)]);
% 
% train_X=train_X(:,1:100);
% train_Y=train_Y(:,1:100);
% train_y=train_y(:,1:100);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [P,h]=EvaluateClassifier(train_X(:,1:20),W,b);
% [ngrad_W, ngrad_b]=ComputeGradsNumSlow(train_X(:,1:20),train_Y(:,1:20),W,b,lambda,1e-5);
% [grad_W,grad_b]=ComputeGradients(train_X(:,1:20),train_Y(:,1:20),P,h,W,b,lambda);
% 
% %Layer 1: 
% Error_W1 = sum(sum(abs(ngrad_W{1} - grad_W{1})/max(eps, sum(sum(abs(ngrad_W{1}) + abs(grad_W{1}))))));
% Error_b1 = sum(abs(ngrad_b{1} - grad_b{1})/max(eps, sum(abs(ngrad_b{1}) + abs(grad_b{1}))));
% 
% %Layer 2:
% Error_W2 = sum(sum(abs(ngrad_W{2} - grad_W{2})/max(eps, sum(sum(abs(ngrad_W{2}) + abs(grad_W{2}))))));
% Error_b2 = sum(abs(ngrad_b{2} - grad_b{2})/max(eps, sum(abs(ngrad_b{2}) + abs(grad_b{2}))));
% 
% fprintf("Layer 1:");
% fprintf("The gradient error of W1: %f \n", Error_W1);
% fprintf("The gradient error of b1: %f \n", Error_b1);
% fprintf("Layer 2:");
% fprintf("The gradient error of W2: %f \n", Error_W2);
% fprintf("The gradient error of b2: %f \n", Error_b2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
eta_min = 1e-5;
eta_max = 1e-1;
n_s = 500;
t = 20000;


% lambda = 10^l;
%lambda = 0.011561;
lambda=0;

GDparams.eta_t = LearningRate(eta_min, eta_max, n_s,t);
GDparams.n_batch = 100;
n_batch = GDparams.n_batch;


N = size(train_X,2);
J_validation = zeros(1, t/N*n_batch);
J_train = zeros(1,t/N*n_batch);
accuracy_train = zeros(1,t/N*n_batch);
accuracy_validation = zeros(1,t/N*n_batch);

% count = 0;  
%  l_min = -5;
%  l_max = -1;
%  l = l_min + (l_max - l_min) * rand(1,1);
%  lambda = 10^l;
for i=1:t/N*n_batch
    
        J_train(i) = ComputeCost(train_X, train_Y, W, b, lambda);
        J_validation(i) = ComputeCost(validation_X, validation_Y, W, b, lambda);
        
        accuracy_train(i) = ComputeAccuracy(train_X,train_y,W,b);
        accuracy_validation(i) = ComputeAccuracy(validation_X,validation_y,W,b);
for j=1:N/n_batch
        j_start = (j-1)*n_batch + 1;
        j_end = j*n_batch;
        inds = j_start:j_end;
        Xbatch = train_X(:, inds);
        Ybatch = train_Y(:, inds);
        k = j + count* N/n_batch;
       
%         eta_t = 0.01;
        eta_t = GDparams.eta_t(k);
        [Wstar,bstar] = MiniBatchGD(Xbatch,Ybatch,W,b,eta_t,lambda);
        W=Wstar;
         b=bstar;
end       

count = count + 1;

end 
figure()
plot (1:t/N*n_batch, J_train, 'g')
hold on
plot (1:t/N*n_batch, J_validation, 'r')
hold off
legend('training cost','validation cost')
legend('training cost')
xlabel('epochs')
ylabel('cost')
% xlabel('update step')
% xticklabels({'0','500','1000','1500','2000','2500','3000','3500','4000','4500','5000'})

figure()
plot (1:t/N*n_batch, accuracy_train, 'b')
hold on
plot (1:t/N*n_batch, accuracy_validation, 'r')
hold off
legend('training accuracy','validation accuracy')
xlabel('epochs')
ylabel('accuracy')
% xlabel('update step')
% xticklabels({'0','500','1000','1500','2000','2500','3000','3500','4000','4500','5000'})

disp(['lambda:' num2str(lambda)])
accuracy_train = ComputeAccuracy(train_X,train_y,W,b);
disp(['Training Accuracy:' num2str(accuracy_train) '%'])

accuracy_test = ComputeAccuracy(testX,testy,W,b);
disp(['Test Accuracy:' num2str(accuracy_test) '%'])


 


% load the data
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

function [W,b] = Initializing_network(d, m, K)

W1 = 1/sqrt(d).*randn(m, d);
W2 = 1/sqrt(m).*randn(K,m);

b1 = zeros(m,1);
b2 = zeros(K,1);

W = {W1, W2};
b = {b1, b2};

end

% P3
function [P,h]=EvaluateClassifier(X, W, b)



s1 = W{1}*X+b{1};
h = max(0,s1);
s = W{2}*h+b{2};
P  = exp(s)./sum(exp(s));
end


function J = ComputeCost(X,Y,W,b,lambda)
    [P,~] = EvaluateClassifier(X,W,b);
    crossentropy_term = sum(diag(-log(double(Y)'*P)));
    reg_term = sum(W{1}(:).^2)+sum(W{2}(:).^2);
    J = (1/(size(X,2))*crossentropy_term)+(lambda*reg_term);     
end

% P5

function acc = ComputeAccuracy(X, y, W, b)

[P,~] = EvaluateClassifier(X, W, b);
[~, ind] = max(P);
acc = length(find(y - ind == 0))/length(y)*100;
end




function [grad_W, grad_b] = ComputeGradients(X, Y, P, h, W, b, lambda)

W1 = W{1};
W2 = W{2};
b1 = b{1};
b2 = b{2};
grad_W1 = zeros(size(W1));
grad_W2 = zeros(size(W2));
grad_b1 = zeros(size(b1));
grad_b2 = zeros(size(b2));

for i = 1 : size(X, 2)
    Pi = P(:, i);
    hi = h(:, i);
    Yi = Y(:, i);
    Xi = X(:, i);
    g = -Yi'*(diag(Pi) - Pi*Pi')/(Yi'*Pi);
    grad_b2 = grad_b2 + g';
    grad_W2 = grad_W2 + g'*hi';
    g = g*W2;
    hi(find(hi > 0)) = 1;
    g = g*diag(hi);
    grad_b1 = grad_b1 + g';
    grad_W1 = grad_W1 + g'*Xi';   
end

grad_W1 = 2*lambda*W1 + grad_W1/size(X, 2);
grad_W2 = 2*lambda*W2 + grad_W2/size(X, 2);
grad_b1 = grad_b1/size(X, 2);
grad_b2 = grad_b2/size(X, 2);
grad_W = {grad_W1, grad_W2}; 
grad_b = {grad_b1, grad_b2};

end





%   [Wstar, bstar] = MiniBatchGD(trainX, trainY, GDparams, W, b, lambda);
function [Wstar,bstar] = MiniBatchGD(X,Y, W,b,eta_t,lambda)
        [P,h] = EvaluateClassifier(X, W, b);
        [grad_W, grad_b] = ComputeGradients(X, Y, P, h, W, b, lambda);
        
        W{1}= W{1} - eta_t * grad_W{1};
        W{2} = W{2} - eta_t * grad_W{2};
        b{1} = b{1} - eta_t * grad_b{1};
        b{2} = b{2} - eta_t * grad_b{2};
        Wstar = W;
        bstar = b;
end 

% ComputeGradsNumSlow
function [grad_W, grad_b] = ComputeGradsNumSlow(X, Y, W, b, lambda, h)

grad_W = cell(numel(W), 1);
grad_b = cell(numel(b), 1);

for j=1:length(b)
    grad_b{j} = zeros(size(b{j}));
    
    for i=1:length(b{j})
        
        b_try = b;
        b_try{j}(i) = b_try{j}(i) - h;
        c1 = ComputeCost(X, Y, W, b_try, lambda);
        
        b_try = b;
        b_try{j}(i) = b_try{j}(i) + h;
        c2 = ComputeCost(X, Y, W, b_try, lambda);
        
        grad_b{j}(i) = (c2-c1) / (2*h);
    end
end

for j=1:length(W)
    grad_W{j} = zeros(size(W{j}));
    
    for i=1:numel(W{j})
        
        W_try = W;
        W_try{j}(i) = W_try{j}(i) - h;
        c1 = ComputeCost(X, Y, W_try, b, lambda);
    
        W_try = W;
        W_try{j}(i) = W_try{j}(i) + h;
        c2 = ComputeCost(X, Y, W_try, b, lambda);
    
        grad_W{j}(i) = (c2-c1) / (2*h);
   end
end
end 

% eta


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

   

