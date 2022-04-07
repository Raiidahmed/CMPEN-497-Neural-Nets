%Raiid Ahmed Midterm Question 3

%Heteroassociative net with 2 bit inputs and 1 bit outputs
%Weights determined using delta rule over 100 epochs

clc
clear

load('CMPEN497_MT_Data_Q3.mat')

S = Training_Data;
t = Target_Data;
STest = Test_Data;
tTest = Test_Target_Data;

epochs = 1:100;
alphastep = (.9 - 1e-5)/(150000-1);
alpha = .9:-alphastep:1e-5;

W = zeros(2,1);
totalerrsq = 0;
MSE = zeros(1,100);
iter = 1;

for x = epochs
    for n = 1:length(S(1,:))
        a = alpha(iter);
        yin = W.' * S(:,n);
        yin = bipolar_act(yin);
        err = t(n) - yin;
        errsq = err^2;
        totalerrsq = totalerrsq + errsq;
        W = W + a * err * S(:,n);
        iter = iter + 1;
    end
    MSE(x) = totalerrsq/(x*n);
end

match = zeros(1,length(STest(1,:)));

for n = 1:length(STest(1,:))
    yout = W.' * STest(:,n);
    yout = bipolar_act(yout);
    if yout == tTest(n)
        match(n) = 1;
    end
end

errorRate = 1 - sum(match)/length(STest(1,:));

decision = @(x) -(W(1)/W(2)).* x;

figure(1)
    plot(MSE)
        xlabel('Epochs')
        ylabel('Mean Squared Error')
        title('MSE vs Epochs')

figure(2)
hold on
    scatter(STest(1,:),STest(2,:))
    fplot(decision)
        xlabel('X1')
        ylabel('X2')
        title('Test Points with Decision Bounduary')
hold off



   
    
    




    

