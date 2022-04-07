%Raiid Ahmed Homework 3 and 4

clc
clear

%Training data here

A = [0,2,0;
     2,0,2;
     2,2,2;
     2,0,2;
     2,0,2];
 
At = [0,0,0];

B = [2,2,0;
     2,0,2;
     2,2,0;
     2,0,2;
     2,2,0];
 
Bt = [0,0,2];

C = [0,2,2;
     2,0,0;
     2,0,0;
     2,0,0;
     0,2,2];
 
Ct = [0,2,0];

D = [2,2,0;
     2,0,2;
     2,0,2;
     2,0,2;
     2,2,0];
 
Dt = [0,2,2];

E = [2,2,2;
     2,0,0;
     2,2,2;
     2,0,0;
     2,2,2];
 
Et = [2,0,0];

F = [2,2,2;
     2,0,0;
     2,2,0;
     2,0,0;
     2,0,0];
 
Ft = [2,0,2];

G = [0,2,2;
     2,0,0;
     2,0,2;
     2,0,2;
     0,2,2];
 
Gt = [2,2,0];

H = [2,0,2;
     2,0,2;
     2,2,2;
     2,0,2;
     2,0,2];
 
Ht = [2,2,2];

%Allocate data in cell arrays
%Remove/add terms in cell array to test different training sets    

SCell = {A,B,C,D,E,F,G,H};
tCell = {At,Bt,Ct,Dt,Et,Ft,Gt,Ht};

%Reshaping arrays into vectors

for x = 1:length(SCell)
    SCell{x} = SCell{x} - 1;
end

for x = 1:length(tCell)
    tCell{x} = tCell{x} - 1;
end

SCellVectorized = {};

for x = 1:length(SCell)
    SCellVectorized{x} = reshape(SCell{x}.',[],1);
end

tCellVectorized = {};

for x = 1:length(tCell)
    tCellVectorized{x} = reshape(tCell{x}.',[],1);
end

%Calculating weight matrix with hebb rule

W = zeros(length(SCellVectorized{x}),length(tCellVectorized{x}));

for x = 1:length(SCellVectorized)
    W = W + SCellVectorized{x} * (tCellVectorized{x}.');
end

SCellinputVectorized = SCellVectorized;

%Uncomment following loop to test noise on S set. Replace .25 with any percentage of
%to test. Noise is randomly generated.

% for x = 1:length(SCellinputVectorized)
%     for i = 1:round(length(SCellinputVectorized{x}) * .25)
%         index = round(rand * length(SCellinputVectorized{x}));
%         if index > length(SCellinputVectorized{x})
%             index = length(SCellinputVectorized{x});
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

tCellinputVectorized = tCellVectorized;

%Uncomment following loop to test noise on t set. Replace .25 with any percentage of
%to test. Noise is randomly generated.
 
% for x = 1:length(tCellinputVectorized)
%     for i = 1:round(length(tCellinputVectorized{x}) * .25)
%         index = round(rand * length(tCellinputVectorized{x}));
%         if index > length(tCellinputVectorized{x})
%             index = length(tCellinputVectorized{x});
%         elseif index < 1
%             index = 1;
%         end
%         noise = rand;
%         if noise < .5
%             noise = -1;
%         elseif noise > .5
%             noise = 1;
%         end
%         tCellinputVectorized{x}(index) = noise;
%     end
% end

%Calculating output activation values using BAM net algorithm

Xi = SCellinputVectorized;
Xdelt = SCellinputVectorized;
Yi = tCellinputVectorized;
Ydelt = tCellinputVectorized;
Xprev = SCellinputVectorized;
Yprev = tCellinputVectorized;

matchT = zeros(1,length(tCellVectorized));
matchS = zeros(1,length(SCellVectorized));

for x = 1:length(SCellinputVectorized)
    while any(Xdelt{x}) && any(Ydelt{x})
        Xprev{x} = Xi{x};
        Yprev{x} = Yi{x};
        for j = 1:length(Yi{x})
            for i = 1:length(Xi{x})
                Yin = sum(Xi{x} .* W(:,j));
                if Yin > 0
                    Yi{x}(j) = 1;
                elseif Yin < 0
                    Yi{x}(j) = -1;
                end
                Xin = sum(Yi{x} .* W(i,:));
                if Xin > 0
                    Xi{x}(i) = 1;
                elseif Xin < 0
                    Xi{x}(i) = -1;
                end
            end
        end
        Xdelt{x} = Xi{x} - Xprev{x};
        Ydelt{x} = Yi{x} - Yprev{x};
    end
    for y = 1:length(tCellVectorized)
        if Yi{x} == tCellVectorized{y}
            matchT(x) = y;
        end
    end
    for y = 1:length(SCellVectorized)
        if Xi{x} == SCellVectorized{y}
            matchS(x) = y;
        end
    end
end

%Displaying match results calculated in iteration loop

disp(matchT)
disp(matchS)
        



