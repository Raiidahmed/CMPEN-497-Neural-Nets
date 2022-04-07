%Raiid Ahmed Homework 3 and 4

clc
clear

%Training data here

A = [0,0,0,2,0,0,0;
     0,0,0,2,0,0,0;
     0,0,0,2,0,0,0;
     0,0,2,0,2,0,0;
     0,0,2,0,2,0,0;
     0,2,0,0,0,2,0;
     0,2,2,2,2,2,0;
     0,2,0,0,0,2,0;
     0,2,0,0,0,2,0];

B = [2,2,2,2,2,2,0;
     2,0,0,0,0,0,2;
     2,0,0,0,0,0,2;
     2,0,0,0,0,0,2;
     2,2,2,2,2,2,0;
     2,0,0,0,0,0,2;
     2,0,0,0,0,0,2;
     2,0,0,0,0,0,2;
     2,2,2,2,2,2,0];
 
C = [0,0,2,2,2,0,0;
     0,2,0,0,0,2,0;
     2,0,0,0,0,0,2;
     2,0,0,0,0,0,0;
     2,0,0,0,0,0,0;
     2,0,0,0,0,0,0;
     2,0,0,0,0,0,2;
     0,2,0,0,0,2,0;
     0,0,2,2,2,0,0];
 
D = [2,2,2,2,2,0,0;
     2,0,0,0,0,2,0;
     2,0,0,0,0,0,2;
     2,0,0,0,0,0,2;
     2,0,0,0,0,0,2;
     2,0,0,0,0,0,2;
     2,0,0,0,0,0,2;
     2,0,0,0,0,2,0;
     2,2,2,2,2,0,0];
 
E = [2,2,2,2,2,2,2;
     2,0,0,0,0,0,0;
     2,0,0,0,0,0,0;
     2,0,0,0,0,0,0;
     2,2,2,2,2,0,0;
     2,0,0,0,0,0,0;
     2,0,0,0,0,0,0;
     2,0,0,0,0,0,0;
     2,2,2,2,2,2,2];

J = [0,0,0,0,0,2,0;
     0,0,0,0,0,2,0;
     0,0,0,0,0,2,0;
     0,0,0,0,0,2,0;
     0,0,0,0,0,2,0;
     0,0,0,0,0,2,0;
     0,2,0,0,0,2,0;
     0,2,0,0,0,2,0;
     0,0,2,2,2,0,0];

K = [2,0,0,0,0,2,0;
     2,0,0,0,2,0,0;
     2,0,0,2,0,0,0;
     2,0,2,0,0,0,0;
     2,2,0,0,0,0,0;
     2,0,2,0,0,0,0;
     2,0,0,2,0,0,0;
     2,0,0,0,2,0,0;
     2,0,0,0,0,2,0];
 
%Allocate data in cell arrays
%Remove/add terms in cell array to test different training sets    

SCell = {A,B,C,D,E,J};

%Reshaping arrays into vectors

for x = 1:length(SCell)
    SCell{x} = SCell{x} - 1;
end

SCellVectorized = {};

for x = 1:length(SCell)
    SCellVectorized{x} = SCell{x}(:);
end

SCellinputVectorized = SCellVectorized;

%Uncomment following loop to test noise. Replace .25 with any percentage of
%to test. Noise is randomly generated.

% for x = 1:length(SCellVectorized)
%     for i = 1:round(length(SCellVectorized{x}) * .25)
%         index = round(rand * length(SCellVectorized{x}));
%         if index > length(SCellVectorized{x})
%             index = length(SCellVectorized{x});
%         elseif index < 1
%             index = 1;
%         end
%         noise = rand;
%         if noise < .5
%             noise = -1;
%         elseif noise > .5
%             noise = 1;
%         end
%         SCellinputVectorized{x}(index) = noise;
%     end
% end

%Calculating weight matrix with hebb rule

W = zeros(length(SCellVectorized{x}));

for x = 1:length(SCellVectorized)
    W = W + SCellVectorized{x} * (SCellVectorized{x}.');
end

W = W - diag(diag(W));

%Calculating output activation values using Hopfield algorithm

order = randperm(63);

Yi = SCellinputVectorized;
Ydelt = SCellinputVectorized;
Yprev = SCellinputVectorized;

match = zeros(1,length(SCellinputVectorized));


for x = 1:length(SCellinputVectorized)
    while any(Ydelt{x}) 
    Yprev{x} = Yi{x};
        for i = order
            Yin = SCellinputVectorized{x}(i) + sum(Yi{x} .* W(:,i));
            if Yin > 0
                Yi{x}(i) = 1;
            elseif Yin < 0
                Yi{x}(i) = -1;
            end
        end
     Ydelt{x} = Yi{x} - Yprev{x};
    end
    for y = 1:length(SCellVectorized)
        if Yi{x} == SCellVectorized{y}
            match(x) = y;
        end
    end
end

%Displaying match results calculated in iteration loop

disp(match)



 


        