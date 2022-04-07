clc
clear
A = [2;
     2;
     2;
     0];

SCell = {A};

for x = 1:length(SCell)
    SCell{x} = SCell{x} - 1;
end

SCellVectorized = {};

for x = 1:length(SCell)
    SCellVectorized{x} = SCell{x}(:);
end

W = zeros(4,4);

for x = 1:length(SCellVectorized)
    W = W + SCellVectorized{x} * (SCellVectorized{x}.');
end

W = W - diag(diag(W));

input = [0,0,1,0].';
Y = [0,0,0,0];
isConverged = false;

while ~isConverged 
    Yprev = Y;
    for x = 1:length(SCell)
        order = [1,4,3,2];
        Yin = input;
        for y = order
            YinRaw = input(y) + sum(Yin.' * W(:,y));
            if YinRaw > Y(y)
                Yin(y) = 1;
            elseif YinRaw < Y(y)
                Yin(y) = 0;
            else
                Yin(y) = Y(y);
            end
        end
        Y = Yin;
    end
    if Yprev == Yin
        isConverged = true;
    end
end
        


 
