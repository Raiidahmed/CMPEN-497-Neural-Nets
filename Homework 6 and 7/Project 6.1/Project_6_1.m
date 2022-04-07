%Raiid Ahmed HW4 Project 6.1

clc
clear

%Allocate test data

S = [1, -1;
    -1,  1;
     1,  1;
    -1, -1];

t = [1; 1; -1; -1];

%Randomize weights

Vinit = rand(3,2) - .5;
Winit = rand(3,1) - .5;
V = Vinit;
W = Winit;

%Allocate epochs and learning rates

epochs = [1000,10000,25000];
alpha = [.05,.25,.5];

%Create containers for unit values

X = zeros(2,1);
Z = zeros(2,1);
Y = 0;
match = zeros(1,4);
sigma = .25;
Zin = zeros(2,1);

%Train and deploy network

for a = alpha
    for e = epochs
        for iter = 1:e
            order = randperm(4);
            for p = order
                X = S(p,:).';
                for j = 1:length(Z)
                    Zin(j) = V(1,j) + sum(X.*V(2:3,j));
                    Z(j) = sigmoid_act(Zin(j),sigma);
                end
                Yin = W(1) + sum(Z.*W(2:3));
                Y = sigmoid_act(Yin,sigma);
                err = (t(p) - Y)*sigmoiddiff_act(Yin,sigma);
                Wdelt(2:3) = a*err.*Z;
                Wdelt(1) = a*err;
                errin = sum(err.*W);
                err = errin.*sigmoiddiff_act(Zin,sigma);
                Vdelt(2:3,:) = a*X*err.';
                Vdelt(1,:) = a*err;
                W = W + Wdelt.';
                V = V + Vdelt;
            end
        end
        for p = 1:4
             X = S(p,:).';
             for j = 1:length(X)
                 Zin = V(1,j) + sum(X.*V(2:3,j));
                 Z(j) = sigmoid_act(Zin,sigma);
             end
             Yin = W(1) + sum(Z.*W(2:3));
             Y = sigmoid_act(Yin,sigma);
             Y = bipolar_act(Y);
             T = t(p);
             if Y == T
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
        Zin = zeros(2,1);
        match = zeros(1,4);
    end
end
                





