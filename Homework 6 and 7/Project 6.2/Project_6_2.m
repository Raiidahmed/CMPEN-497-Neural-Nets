%Raiid Ahmed HW4 Project 6.2

clc
clear

%Allocate test data

S = [0,1,2,3,4,5,6,7,8,9;
     9,8,7,6,5,4,3,2,1,0;
     0,9,1,8,2,7,3,6,4,5;
     4,5,6,3,2,7,1,8,0,9;
     3,8,2,7,1,6,0,5,9,4;
     1,6,0,7,4,8,3,9,2,5;
     2,1,3,0,4,9,5,8,6,7;
     9,4,0,5,1,6,2,7,3,8];

t = [-1,-1;
      1, 1;
     -1, 1;
     -1, 1;
      1,-1;
      1, 1;
     -1, 1;
     -1,-1];
 
%Randomize weights

Vinit = rand(11,4) - .5;
Winit = rand(5,2) - .5;
Wdelt = zeros(5,2);
Vdelt = zeros(11,4);
V = Vinit;
W = Winit;

%Allocate epochs and learning rates

epochs = [5000,50000];
alpha = [.05,.5];

%Create containers for unit values

X = zeros(10,1);
Z = zeros(4,1);
Y = zeros(2,1);
match = zeros(1,8);
sigma = .25;
Zin = zeros(4,1);
Yin = zeros(2,1);
err = zeros(1,2);
errin = zeros(1,4);
err2 = zeros(1,4);

%Training and analysis

for a = alpha
    for e = epochs
        for iter = 1:e
            order = randperm(8);
            for p = order
                X = S(p,:).';
                for j = 1:length(Z)
                    Zin(j) = V(1,j) + sum(X.*V(2:11,j));
                    Z(j) = sigmoid_act(Zin(j),sigma);
                end
                for k = 1:length(Y)
                    Yin(k) = W(1,k) + sum(Z.*W(2:5,k));
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
                 Zin(j) = V(1,j) + sum(X.*V(2:11,j));
                 Z(j) = sigmoid_act(Zin(j),sigma);
             end
             for k = 1:length(Y)
                 Yin(k) = W(1,k) + sum(Z.*W(2:5,k));
                 Y(k) = sigmoid_act(Yin(k),sigma);
                 Y(k) = bipolar_act(Y(k));
             end
             T = t(p,:);
             if Y(1) == T(1) && Y(2) == T(2)
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
        disp("Pattern Matches: ")
        disp(match)
        V = Vinit;
        W = Winit;
        match = zeros(1,8);
    end
end







