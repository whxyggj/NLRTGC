
clear
projev = 1.5;

%% BBC
dataset='bbcsport_2view.mat';
numClust=5;
num_views=2;
load(dataset);
X1=data{1};
X2=data{2};
sigma(1)=optSigma(X1);
sigma(2)=optSigma(X2);

gt=truth;
cls_num = length(unique(gt));
tic
%% Construct kernel and transition matrix
K=[];
T=cell(1,num_views);
for j=1:num_views
    options.KernelType = 'Gaussian';
    options.t = sigma(j);
    K(:,:,j) = constructKernel(data{j},data{j},options);
    D=diag(sum(K(:,:,j),2));
    L_rw=D^-1*K(:,:,j);
    T{j}=L_rw;
end
T_tensor = cat(3, T{:,:});
t = T_tensor(:);



%% init evaluation result
best_single_view.nmi=0;
feature_concat.nmi=0;
kernel_addition.nmi=0;
markov_mixture.nmi=0;
co_reg.nmi=0;
markov_ag.nmi=0;

V = length(data); 
N = size(data{1},1); % number of samples

for k=1:V
    Z{k} = zeros(N,N);
    Y{k} = zeros(N,N);
    E{k} = zeros(N,N); 
    U{k} = zeros(N,N);
    FI{k} = zeros(N,N);
    PI{k} = zeros(N,N);
    QI{k} = zeros(N,N);
end
Z_tensor = cat(3, Z{:,:});
E_tensor = cat(3, E{:,:});
w=ones(1,3)*(V);
hv = zeros(1,3);
hvs = 0;

%% initial L
for k = 1:V
    D = diag(sum(T{k}));
    L{k} = D - T{k};
end

pi = zeros(N*N*V,1);
dim1 = N;dim2 = N;dim3 = V;
sX = [N, N, V];

tol = 1e-6;
lambda1 = 0.03;
lambda2 = 0.1;
lambda3=0.01;
alfa = 1.2;
theta = 1;

iter = 0;
mu = 10e-3; 
max_mu = 10e10; 
pho_mu = 2;
max_iter=200;

while iter < max_iter
    Zpre=Z_tensor;
    Epre=E_tensor;
    fprintf('----processing iter %d--------\n', iter+1);
    %% update Z
    for k= 1:V
        tmp = T{k}-E{k}+FI{k}/mu+Y{k}-PI{k}/mu+U{k}-QI{k}/mu;
        Z{k}=tmp/(3*eye(N,N)+lambda2/mu*(L{k}+L{k}'));
    end
    for k = 1:V
        D = diag(sum(Z{k}));
        L{k} = D - Z{k};
    end

    %% update Z*
    Z_hat = zeros(N,N);
    w_hat = 0;
    for k=1:V
        Z_hat = Z_hat+w(k)*Z{k};
        w_hat = w_hat+w(k);
    end
    Z_hat= Z_hat/w_hat;

    %% update Y
    Z_tensor = cat(3, Z{:,:});
    z = Z_tensor(:);
    PI_tensor = cat(3, PI{:,:});
    pi = PI_tensor(:);

    [y, ~] = wshrinkObj_tanh(z + 1/mu*pi,1/mu,sX,0,3,alfa,theta)   ;
    Y_tensor = reshape(y, sX);
    Y{1}=Y_tensor(:,:,1);
    Y{2}=Y_tensor(:,:,2);

    %% update E
    EF = [T{1}-Z{1}+FI{1}/mu;T{2}-Z{2}+FI{2}/mu];
    [Econcat] = solve_l1l2(EF,lambda1/mu);

    E{1} = Econcat(1:size(T{1},1),:);
    E{2} = Econcat(size(T{1},1)+1:size(T{1},1)+size(T{2},1),:);

    E_tensor = cat(3, E{:,:});

    %% update w
    for k=1:V
        hv(k) = norm(U{k}-Z_hat,'fro');
        hvs = hvs + hv(k);
    end
    for k=1:V
        w(k)=hv(k)/hvs;
    end
    hvs = 0;

    %% update U
    for k=1:V
        U{k}=(2*lambda3/w(k)*Z_hat+mu*Z{k}+QI{k})/(2*lambda3/w(k)+mu);
    end
    %% update FI,PI
    for k=1:V
        FI{k} = FI{k} + mu*(T{k}-Z{k}-E{k});
        PI{k} = PI{k} + mu*(Z{k}-Y{k});
        QI{k} = QI{k} + mu*(Z{k}-U{k});
    end

    %% check convergence
    leq = T_tensor-Z_tensor-E_tensor;
    leqm = max(abs(leq(:)));
    difZ = max(abs(Z_tensor(:)-Zpre(:)));
    difE = max(abs(E_tensor(:)-Epre(:)));
    err = max([leqm,difZ,difE]);
    fprintf('iter = %d, mu = %.3f, difZ = %.3f, difE = %.8f,err=%d\n'...
            , iter,mu,difZ,difE,err);
    if err < tol
        break;
    end

    iter = iter + 1;
    mu = min(mu*pho_mu, max_mu);
end
toc
time_cost = toc;
S = zeros(N,N);
for k=1:num_views
    S = S + Z{k};
end
[pi,~]=eigs(S',1);
Dist=pi/sum(pi);
pi=diag(Dist);
P_hat=(pi^0.5*S*pi^-0.5+pi^-0.5*S'*pi^0.5)/2;
for i=1:10
    C{i} = SpectralClustering(S,cls_num);
    [Fi(i),Pi(i),Ri(i)] = compute_f(truth,C{i});
    ACCi(i) = Accuracy(C{i},double(truth));
    [A nmii(i) avgenti(i)] = compute_nmi(truth,C{i});
    if (min(truth)==0)
        [ARi(i),RIi(i),MIi(i),HIi(i)]=RandIndex(truth+1,C{i});
    else
        [ARi(i),RIi(i),MIi(i),HIi(i)]=RandIndex(truth,C{i});
    end       
end
ACC(1)=mean(ACCi);ACC(2) = std(ACCi);
nmi(1) = mean(nmii); nmi(2) = std(nmii);
AR(1) = mean(ARi);AR(2) = std(ARi);
F(1) = mean(Fi); F(2) = std(Fi);
P(1) = mean(Pi); P(2) = std(Pi);
R(1) = mean(Ri); R(2) = std(Ri);
avgent(1) = mean(avgenti); avgent(2) = std(avgenti);
result.ACC(1)=mean(ACCi); result.ACC(2) = std(ACCi);
result.nmi(1) = mean(nmii); result.nmi(2) = std(nmii);
result.AR(1) = mean(ARi); result.AR(2) = std(ARi);
result.F(1) = mean(Fi); result.F(2) = std(Fi);
result.P(1) = mean(Pi); result.P(2) = std(Pi);
result.R(1) = mean(Ri); result.R(2) = std(Ri);
result


