%Raiid Ahmed Homework 3 and 4

clc
clear

%Training data here

One = [0,0,2,0,0;
       0,2,2,0,0;
       0,0,2,0,0;
       0,0,2,0,0;
       0,0,2,0,0;
       0,0,2,0,0;
       0,2,2,2,0];
   
Two = [0,2,2,2,0;
       2,0,0,0,2;
       0,0,0,0,2;
       0,0,0,2,0;
       0,0,2,0,0;
       0,2,0,0,0;
       2,2,2,2,2];
   
Three = [0,2,2,2,0;
         2,0,0,0,2;
         0,0,0,0,2;
         0,0,2,2,0;
         0,0,0,0,2;
         2,0,0,0,2;
         0,2,2,2,0];
     
Four = [0,0,0,0,2;
        0,0,0,2,2;
        0,0,2,0,2;
        0,2,0,0,2;
        2,2,2,2,2;
        0,0,0,0,2;
        0,0,0,0,2];

Five = [2,2,2,2,2;
        2,0,0,0,0;
        2,0,0,0,0;
        2,2,2,2,0;
        0,0,0,0,2;
        0,0,0,0,2;
        2,2,2,2,0];
    
Six = [0,2,2,2,2;
       2,0,0,0,0;
       2,0,0,0,0;
       2,2,2,2,0;
       2,0,0,0,2;
       2,0,0,0,2;
       0,2,2,2,0];

Seven = [2,2,2,2,2;
         0,0,0,0,2;
         0,0,0,2,0;
         0,0,0,2,0;
         0,0,2,0,0;
         0,0,2,0,0;
         0,2,0,0,0];

Eight = [2,2,2,2,0;
         2,0,0,0,2;
         2,0,0,0,2;
         0,2,2,2,0;
         2,0,0,0,2;
         2,0,0,0,2;
         0,2,2,2,0];

Nine = [0,2,2,2,0;
        2,0,0,0,2;
        2,0,0,0,2;
        0,2,2,2,2;
        0,0,0,0,2;
        0,0,0,0,2;
        2,2,2,2,0];

Zero = [0,2,2,2,0;
        2,0,0,0,2;
        2,0,0,0,2;
        2,0,0,0,2;
        2,0,0,0,2;
        2,0,0,0,2;
        0,2,2,2,0];

%Allocate data in cell arrays
%Remove/add terms in cell array to test different training sets
 
SCell = {One,Two,Three,Four,Five,Six,Seven,Eight,Nine,Zero};

%Reshaping arrays into vectors

for x = 1:length(SCell)
    SCell{x} = SCell{x} - 1;
end

SCellVectorized = {};

for x = 1:length(SCell)
    SCellVectorized{x} = SCell{x}(:);
end

%Calculating weight matrix with hebb rule

W = zeros(length(SCellVectorized{x}));

for x = 1:length(SCellVectorized)
    W = W + SCellVectorized{x} * (SCellVectorized{x}.');
end

W = W - diag(diag(W));

SCellout = {};

%Designating input set

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

%Calculating outputs

for x = 1:length(SCellinputVectorized)
    SCellout{x} = (W.') * SCellinputVectorized{x};
end

SCelloutBinarized = {};

%Passing activation function

for x = 1:length(SCellout)
    for y = 1:length(SCellout{x})
        if SCellout{x}(y) > 0
            SCelloutBinarized{x}(y,1) = 1;
        else 
            SCelloutBinarized{x}(y,1) = -1;
        end
    end
end

%Reorganizing output into matrix form

SCelloutBinarizedUnvectorized = {};

for x = 1:length(SCelloutBinarized)
    SCelloutBinarizedUnvectorized{x} = [SCelloutBinarized{x}(1:7),SCelloutBinarized{x}(8:14),SCelloutBinarized{x}(15:21),SCelloutBinarized{x}(22:28),SCelloutBinarized{x}(29:35)];
end

%Checking for correct matches

match = zeros(1,length(SCelloutBinarizedUnvectorized));

for x = 1:length(SCelloutBinarizedUnvectorized)
    for y = 1:length(SCell)
        if SCelloutBinarizedUnvectorized{x} == SCell{y}
            match(x) = y;
        end
    end
end

disp(match)
