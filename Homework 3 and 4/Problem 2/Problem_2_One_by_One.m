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

SCell = {One,Two,Four,Six,Seven};

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


SCellinputVectorized = SCellVectorized;

%Uncomment following loop to test noise. Replace .25 with any percentage of
%to test. Noise is randomly generated.

for x = 1:length(SCellVectorized)
    for i = 1:round(length(SCellVectorized{x}) * .10)
        index = round(rand * length(SCellVectorized{x}));
        if index > length(SCellVectorized{x})
            index = length(SCellVectorized{x});
        elseif index < 1
            index = 1;
        end
        noise = rand;
        if noise < .5
            noise = -1;
        elseif noise > .5
            noise = 1;
        end
        SCellinputVectorized{x}(index) = noise;
    end
end

%Calculating output activation values using one by one iteration mode

order = randperm(35);
Xi = SCellinputVectorized;
Cellout = SCellinputVectorized;
Xdelt = SCellinputVectorized;
Xprev = SCellinputVectorized;
match = zeros(1,length(SCellinputVectorized));

for x = 1:length(SCellinputVectorized)
    while any(Xdelt{x})
        Xprev{x} = Xi{x};
        for i = order
            Xin = SCellinputVectorized{x}(i)+ sum(Xi{x} .* W(:,i));
            if Xin > 0
                Xi{x}(i) = 1;
            elseif Xin < 0
                Xi{x}(i) = -1;
            end
        end
        Xdelt{x} = Xi{x} - Xprev{x};
        Cellout{x} = Xi{x};
    end
    for y = 1:length(SCellVectorized)
        if Xi{x} == SCellVectorized{y}
            match(x) = y;
        end
    end 
end

%Displaying match results calculated in iteration loop

disp(match)

