%Raiid Ahmed Homework 5 Project 4.1
clc
clear

%Initialize training set

x = zeros(100,2);
iter = 1;

while iter <= 100
    x1 = rand - .5;
    x2 = rand - .5;

    if (x1^2 + x2^2) < .25
        x(iter,1) = x1;
        x(iter,2) = x2;
        iter = iter + 1;
    end
end

%Initialize weights

w = rand(2,50);
w = w * 2 - 1;

%Initialize learning rates

alpha_step = (.5-.01)/99;
alpha = .5:-alpha_step:.01;

%Initialize radius

R = 1;

%Training Algorithm

D = zeros(1,length(w(1,:)));

subplot(3,4,1)
%figure(epoch/10)
    plot(w(1,:),w(2,:))
        title('Initial')
        xlabel('W(1,j)')
        ylabel('W(2,j)')

for epoch = 1:100
    a = alpha(epoch);
    order = randperm(length(x(:,1)));
        for input = order
            x_in = x(input,:).';
            
            for j = 1:length(w(1,:))
                D(j) = sum(((w(:,j) - x_in).^2));
            end
            
            [Winner_Val, J] = min(D);

            w(:,J) = w(:,J) + a*(x_in - w(:,J));

            if (J + R) <= 50
                w(:,(J + R)) = w(:,(J + R)) + a*(x_in - w(:,(J + R)));
            end

            if (J - R) >= 1
                w(:,(J - R)) = w(:,(J - R)) + a*(x_in - w(:,(J - R)));
            end
        end
    if mod(epoch,10) == 0
        subplot(3,4,(epoch/10) + 1)
        %figure(epoch/10)
            plot(w(1,:),w(2,:))
                title("Epoch " + num2str(epoch))
                xlabel('W(1,j)')
                ylabel('W(2,j)')
    end
end










