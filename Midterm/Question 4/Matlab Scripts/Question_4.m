%Raiid Ahmed Midterm Question 4

%Kohonen 1-D Lattice
%2-D Randomized inputs

clc
clear

load('CMPEN497_MT_Data_Q4.mat')

%Initialize training set

x = SOM_Data;

%Initialize weights

w = rand(2,50);
w = w * .5 + .25;

%Initialize learning rates

alpha_step = (.5-.01)/99;
alpha = .5:-alpha_step:.01;

%Initialize radii

R_step = (5 - 1)/4999;
Radius = 5:-R_step:1;

D = zeros(1,length(w(1,:)));
iter_alpha = 1;
iter_R = 1;

%Intial plot

iter_plot = 1;
subplot(2,3,iter_plot)
    plot(w(1,:),w(2,:))
        title('Initial')
        xlabel('W(1,j)')
        ylabel('W(2,j)')
        
%Training algorithm

for epoch = 1:50
    order = randperm(length(x(:,1)));
        R = round(Radius(iter_R));
        for input = order
            a = alpha(iter_alpha);
            x_in = x(input,:).';
            
            for j = 1:length(w(1,:))
                D(j) = sum(((w(:,j) - x_in).^2));
            end
            
            [Winner_Val, J] = min(D);

            w(:,J) = w(:,J) + a*(x_in - w(:,J));
            
            for neighbor = (J - R):(J + R)
                if neighbor <= 50 && neighbor >= 1
                    w(:,(neighbor)) = w(:,(neighbor)) + a*(x_in - w(:,(neighbor)));
                end
            end
            
            iter_alpha = iter_alpha + 1;
            iter_R = iter_R + 1;
            
            %Plotting
            
            if iter_R == 10 || iter_R == 50 || iter_R == 100 || iter_R == 1000 || iter_R == 5000
                iter_plot = iter_plot + 1; 
                subplot(2,3,iter_plot)
                    plot(w(1,:),w(2,:))
                    title("Iteration " + num2str(iter_R))
                    xlabel('W(1,j)')
                    ylabel('W(2,j)')
            end
        end
    iter_alpha = 1;
end







