%Raiid Ahmed HW4 Project 6.4

clc
clear

%Allocate training data

funct = @(x1,x2) sin(2.*pi.*x1).*sin(2.*pi.*x2);
X1 = 0:.2:1;
X2 = 0:.2:1;
iter = 1;

for i = 1:length(X1)
    for j = 1:length(X2)
        t(iter) = funct(X1(i),X2(j));
        S(iter,1) = X1(i);
        S(iter,2) = X2(j);
        iter = iter + 1;
    end
end

%Allocate test data

X1t = 0:.1:1;
X2t = 0:.1:1;
iter = 1;

for i = 1:length(X1t)
    for j = 1:length(X2t)
        ttest(iter) = funct(X1t(i),X2t(j));
        Stest(iter,1) = X1t(i);
        Stest(iter,2) = X2t(j);
        iter = iter + 1;
    end
end

X1tspace = reshape([NaN(size(X1t)); X1t], 1, []);
t = t.';
ttest = ttest.';
patterns = length(t);
testpatterns = length(ttest);

%Select hidden nodes

hnodes = 10;
 
%Randomize weights

Vinit = rand(3,hnodes) - .5;
Winit = rand(hnodes+1,1) - .5;
Wdelt = zeros(hnodes+1,1);
Vdelt = zeros(3,hnodes);
V = Vinit;
W = Winit;

%Allocate epochs and learning rates

epochs = [1000,10000];
alpha = .25;

%Create containers for unit values

X = zeros(2,1);
Z = zeros(hnodes,1);
Y = zeros(1,1);
match = zeros(length(X1t)*2+1,length(X1t)+1);
sigma = .25;
Zin = zeros(hnodes,1);
Yin = zeros(1,1);
err = zeros(1,1);
errin = zeros(1,hnodes);
err2 = zeros(1,hnodes);
Ycoord = zeros(1,length(ttest));

%Training and analysis

for a = alpha
    for e = epochs
        for iter = 1:e
            order = randperm(patterns);
            for p = order
                X = S(p,:).';
                for j = 1:length(Z)
                    Zin(j) = V(1,j) + sum(X.*V(2:3,j));
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
        for p = 1:testpatterns
             X = Stest(p,:).';
             for j = 1:length(Z)
                 Zin(j) = V(1,j) + sum(X.*V(2:3,j));
                 Z(j) = sigmoid_act(Zin(j),sigma);
             end
             for k = 1:length(Y)
                 Yin(k) = W(1,k) + sum(Z.*W(2:hnodes+1,k));
                 Y(k) = sigmoid_act(Yin(k),sigma);
             end
             Ycoord(p) = Y;
        end
        match(1:length(X1tspace),1) = flip(X1tspace.');
        match(length(X2t)*2+1,2:length(X2t)+1) = X2t;
        for i = 1:length(X1t)
            match(length(match(:,1))-i*2,2:length(match(1,:))) = ttest(length(X1t)*(i-1)+1:length(X1t)*i);
        end
        for i = 1:length(X1t)
            match(length(match(:,1))-((i-1)*2+1),2:length(match(1,:))) = Ycoord(length(X1t)*(i-1)+1:length(X1t)*i);
        end
        figure(1)
            plot3(Stest(:,1),Stest(:,2),ttest,'.','MarkerSize',25,'Color','b')
                title('Real Results')
                    xlabel('X1')
                    ylabel('X2')
                    zlabel('Y')
        figure(e)
            plot3(Stest(:,1),Stest(:,2),Ycoord,'.','MarkerSize',25,'Color','r')
                title("Network Results for " + e + " epochs")
                    xlabel('X1')
                    ylabel('X2')
                    zlabel('Y')
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
        disp("Pattern Matches: ")
        disp(match)
        writematrix(match, num2str(e) + ".csv")
        match = zeros(length(X1)*2,length(X1));
        V = Vinit;
        W = Winit;
    end
end

