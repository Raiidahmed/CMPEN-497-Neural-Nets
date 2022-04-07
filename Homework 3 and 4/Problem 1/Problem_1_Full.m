%Raiid Ahmed Homework 3 and 4

clc
clear

%Training data here

A = [0,0,0,2,0,0,0;
     0,0,0,2,0,0,0;
     0,0,2,0,2,0,0;
     0,0,2,0,2,0,0;
     0,0,2,2,2,0,0;
     0,2,0,0,0,2,0;
     0,2,0,0,0,2,0;
     2,0,0,0,0,0,2;
     2,0,0,0,0,0,2];
 
Ay = [0,2,0;
      2,0,2;
      2,2,2;
      2,0,2;
      2,0,2];

B = [2,2,2,2,2,0,0;
     2,0,0,0,0,2,0;
     2,0,0,0,0,0,2;
     2,0,0,0,0,2,0;
     2,2,2,2,2,0,0;
     2,0,0,0,0,2,0;
     2,0,0,0,0,0,2;
     2,0,0,0,0,2,0;
     2,2,2,2,2,0,0];
 
By = [2,2,0;
      2,0,2;
      2,2,0;
      2,0,2;
      2,2,0];
  
C = [0,0,2,2,2,0,0;
     0,2,0,0,0,2,0;
     2,0,0,0,0,0,2;
     2,0,0,0,0,0,0;
     2,0,0,0,0,0,0;
     2,0,0,0,0,0,0;
     2,0,0,0,0,0,2;
     0,2,0,0,0,2,0;
     0,0,2,2,2,0,0];

Cy = [2,2,2;
      2,0,0;
      2,0,0;
      2,0,0;
      2,2,2];
  
D = [2,2,2,2,2,0,0;
     2,0,0,0,0,2,0;
     2,0,0,0,0,0,2;
     2,0,0,0,0,0,2;
     2,0,0,0,0,0,2;
     2,0,0,0,0,0,2;
     2,0,0,0,0,0,2;
     2,0,0,0,0,2,0;
     2,2,2,2,2,0,0];
 
Dy = [2,2,0;
      2,0,2;
      2,0,2;
      2,0,2;
      2,2,0];

E = [2,2,2,2,2,2,2;
     2,0,0,0,0,0,0;
     2,0,0,0,0,0,0;
     2,0,0,0,0,0,0;
     2,2,2,2,2,0,0;
     2,0,0,0,0,0,0;
     2,0,0,0,0,0,0;
     2,0,0,0,0,0,0;
     2,2,2,2,2,2,2];

Ey = [2,2,2;
      2,0,0;
      2,2,2;
      2,0,0;
      2,2,2];

J = [0,0,0,0,0,2,0;
     0,0,0,0,0,2,0;
     0,0,0,0,0,2,0;
     0,0,0,0,0,2,0;
     0,0,0,0,0,2,0;
     0,0,0,0,0,2,0;
     0,2,0,0,0,2,0;
     0,2,0,0,0,2,0;
     0,0,2,2,2,0,0];
 

Jy = [0,0,2;
      0,0,2;
      0,0,2;
      2,0,2;
      0,2,0];
  
K = [2,0,0,0,0,2,0;
     2,0,0,0,2,0,0;
     2,0,0,2,0,0,0;
     2,0,2,0,0,0,0;
     2,2,0,0,0,0,0;
     2,0,2,0,0,0,0;
     2,0,0,2,0,0,0;
     2,0,0,0,2,0,0;
     2,0,0,0,0,2,0];

Ky = [2,0,2;
      2,2,0;
      2,0,0;
      2,2,0;
      2,0,2];
  
%Allocate data in cell arrays
%Remove/add terms in cell array to test different training sets

SCell = {A,B,C,D,E,J,K};
tCell = {Ay,By,Cy,Dy,Ey,Jy,Ky};

%Reshaping arrays into vectors

for x = 1:length(SCell)
    SCell{x} = SCell{x} - 1;
end

for x = 1:length(tCell)
    tCell{x} = tCell{x} - 1;
end

for x = 1:length(SCell)
    SCellVectorized{x} = SCell{x}(:);
end

for x = 1:length(tCell)
    tCellVectorized{x} = tCell{x}(:);
end

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

%Calculating weights with hebb rule

W = zeros(length(SCellVectorized{x}),length(tCellVectorized{x}));

for x = 1:length(SCellVectorized)
    W = W + SCellVectorized{x} * (tCellVectorized{x}.');
end
 
tCellout = {};

%Calculating outputs 

for x = 1:length(SCellVectorized)
    tCellout{x} = (W.') * SCellinputVectorized{x};
end

%Passing activation function

tCelloutBinarized = {};

for x = 1:length(tCellout)
    for y = 1:length(tCellout{x})
        if tCellout{x}(y) > 0
            tCelloutBinarized{x}(y,1) = 1;
        else 
            tCelloutBinarized{x}(y,1) = -1;
        end
    end
end

%Reorganizing output into matrix form

tCelloutBinarizedUnvectorized = {};

for x = 1:length(tCelloutBinarized)
    tCelloutBinarizedUnvectorized{x} = [tCelloutBinarized{x}(1:5),tCelloutBinarized{x}(6:10),tCelloutBinarized{x}(11:15)];
end

match = zeros(1,length(tCelloutBinarizedUnvectorized));

%Checking for correct matches

for x = 1:length(tCelloutBinarizedUnvectorized)
    for y = 1:length(tCell)
        if tCelloutBinarizedUnvectorized{x} == tCell{y}
            match(x) = y;
        end
    end
end

disp(match)


