%Raiid Ahmed HW4 Project 6.2

clc
clear

%Allocate test data

A = [0,2,0;
     2,0,2;
     2,2,2;
     2,0,2;
     2,0,2];
 
At = [0,0,0];

B = [2,2,0;
     2,0,2;
     2,2,0;
     2,0,2;
     2,2,0];
 
Bt = [0,0,2];

C = [0,2,2;
     2,0,0;
     2,0,0;
     2,0,0;
     0,2,2];
 
Ct = [0,2,0];

D = [2,2,0;
     2,0,2;
     2,0,2;
     2,0,2;
     2,2,0];
 
Dt = [0,2,2];

E = [2,2,2;
     2,0,0;
     2,2,2;
     2,0,0;
     2,2,2];
 
Et = [2,0,0];

F = [2,2,2;
     2,0,0;
     2,2,0;
     2,0,0;
     2,0,0];
 
Ft = [2,0,2];

G = [0,2,2;
     2,0,0;
     2,0,2;
     2,0,2;
     0,2,2];
 
Gt = [2,2,0];

H = [2,0,2;
     2,0,2;
     2,2,2;
     2,0,2;
     2,0,2];
 
Ht = [2,2,2];

S = [reshape(A.',1,[]);
     reshape(B.',1,[]);
     reshape(C.',1,[]);
     reshape(D.',1,[]);
     reshape(E.',1,[]);
     reshape(F.',1,[]);
     reshape(G.',1,[]);
     reshape(H.',1,[])];

t = [At;Bt;Ct;Dt;Et;Ft;Gt;Ht];

S = S - 1;
t = t - 1;

%Select hidden nodes and weight ranges

hnodes = 4;
range = .1;

%Randomize weights

Vinit = rand(16,hnodes).*(range*2) - ((range*2)/2);
Winit = rand(hnodes+1,3).*(range) - (range/2);
Wdelt = zeros(hnodes+1,3);
Vdelt = zeros(16,hnodes);
V = Vinit;
W = Winit;

%Allocate epochs and learning rates

epochs = 25000;
alpha = [.05,.25,.5];

%Create containers for unit values

X = zeros(15,1);
Z = zeros(hnodes,1);
Y = zeros(3,1);
match = zeros(1,8);
sigma = .25;
Zin = zeros(hnodes,1);
Yin = zeros(3,1);
err = zeros(1,3);
errin = zeros(1,hnodes);
err2 = zeros(1,hnodes);

for a = alpha
    for e = epochs
        for iter = 1:e
            order = randperm(8);
            for p = order
                X = S(p,:).';
                for j = 1:length(Z)
                    Zin(j) = V(1,j) + sum(X.*V(2:16,j));
                    Z(j) = sigmoid_act(Zin(j),sigma);
                end
                for k = 1:length(Y)
                    Yin(k) = W(1,k) + sum(Z.*W(2:hnodes+1,k));
                    Y(k) = sigmoid_act(Yin(k),sigma);
                end
                for k = 1:length(Y)
                    err(k) = (t(p,k) - Y(k)).*sigmoiddiff_act(Yin(k),sigma);
                end
                for j = 1:length(Z)
                    for k = 1:length(Y)
                        Wdelt(j+1,k) = a*err(k).*Z(j);
                    end
                end
                Wdelt(1,:) = a*err;
                for j = 1:length(Z)
                    errin(j) = sum(err.*W(j+1,:));
                    err2(j) = errin(j).*sigmoiddiff_act(Zin(j),sigma);
                end
                for i = 1:length(X)
                    for j = 1:length(Z)
                        Vdelt(i+1,j) = a*X(i)*err2(j).';
                    end
                end
                Vdelt(1,:) = a*err2;
                W = W + Wdelt;
                V = V + Vdelt;
            end
        end
        for p = 1:8
             X = S(p,:).';
             for j = 1:length(Z)
                 Zin(j) = V(1,j) + sum(X.*V(2:16,j));
                 Z(j) = sigmoid_act(Zin(j),sigma);
             end
             for k = 1:length(Y)
                 Yin(k) = W(1,k) + sum(Z.*W(2:hnodes+1,k));
                 Y(k) = sigmoid_act(Yin(k),sigma);
                 Y(k) = bipolar_act(Y(k));
             end
             T = t(p,:);
             if Y(1) == T(1) && Y(2) == T(2) && Y(3) == T(3)
                 match(p) = 1;
             end
        end
        disp("V Initial: ")
        disp(Vinit)
        disp("W Initial: ")
        disp(Winit)
        disp("V Weights: ")
        disp(V)
        disp("W Weights: ")
        disp(W)
        disp("Learning Rate: " + a)
        disp("Epochs: " + e)
        disp("# of hidden units: " + hnodes)
        disp("Initial Weight Range: " + range + " to -" + range)
        disp("Pattern Matches: ")
        disp(match)
        V = Vinit;
        W = Winit;
        match = zeros(1,8);
    end
end



