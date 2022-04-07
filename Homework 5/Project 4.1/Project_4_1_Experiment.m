%Raiid Ahmed Homework 5 Project 4.1 Experiment

%Changes from original script:
%Reduce input set to a single 2 element vector
%Run net for 1000000 epochs, with vector changing each time
%Plotting map after every run of 1000000 epochs
%Takes at least a few minutes to run and display plots

clc
clear

for z = 1:5 
    %Initialize training set

%     x = zeros(100,2);
%     iter = 1;
% 
%     while iter <= 100
%         x1 = rand - .5;
%         x2 = rand - .5;
% 
%         if (x1^2 + x2^2) < .25
%             x(iter,1) = x1;
%             x(iter,2) = x2;
%             iter = iter + 1;
%         end
%     end

    %Initialize weights

    w = rand(2,50);
    w = w * 2 - 1;

    %Initialize learning rates

    alpha_step = (.5-.01)/999999;
    alpha = .5:-alpha_step:.01;

    %Initialize radius

    R = 1;

    %Training Algorithm

    D = zeros(1,length(w(1,:)));

    % subplot(11,1,1)
    %     plot(w(1,:),w(2,:))

    for epoch = 1:1000000
        a = alpha(epoch);
        check = false;
        while check == false
            x1 = rand - .5;
            x2 = rand - .5;

            if (x1^2 + x2^2) < .25
                x_in = [x1;
                        x2];
                check = true;
            end
        end                

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
    %     if mod(epoch,10) == 0
    %         subplot(11,1,(epoch/10) + 1)
    %             plot(w(1,:),w(2,:))
    %     end
    
    
    subplot(3,2,z)
        plot(w(1,:),w(2,:))
            title("Run " + num2str(z))
            xlabel('W(1,j)')
            ylabel('W(2,j)')
end










