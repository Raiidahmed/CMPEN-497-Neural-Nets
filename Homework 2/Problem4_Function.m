%inputs: initial weights, t, alpha
%outputs: changed weights, y

function [weightsout, y, iserror] = Problem4_Function(input, weightsin, t, alpha)

    z1 = input*weightsin(1,1:2).' + weightsin(1,3) %output not suppressed for debugging
    z2 = input*weightsin(2,1:2).' + weightsin(2,3)
    weightsout = weightsin;
   
    if z1 < 0 && z2 < 0 
        y = -1;
    else
        y = 1;
    end
    
    if y == t
        iserror = false;
    else
        iserror = true;
    end
    
    if y ~= t
        if t == 1
            Zmin = min([abs(z1) abs(z2)]);
            if Zmin == abs(z1)
                for j = 1:length(weightsin(1,1:2))
                    weightsout(1,j) = weightsin(1,j) + alpha * (t - z1) * input(j);
                end
                    weightsout(1,3) = weightsin(1,3) + alpha * (t - z1);
            end
            if Zmin == abs(z2)
                for j = 1:length(weightsin(2,1:2))
                    weightsout(2,j) = weightsin(2,j) + alpha * (t - z2) * input(j);
                end
                    weightsout(2,3) = weightsin(2,3) + alpha * (t - z2);
            end
        else 
            if z1 > 0
                for j = 1:length(weightsin(1,1:2))
                    weightsout(1,j) = weightsin(1,j) + alpha * (t - z1) * input(j);
                end
                weightsout(1,3) = weightsin(1,3) + alpha * (t - z1);
            end
            if z2 > 0
                for j = 1:length(weightsin(2,1:2))
                    weightsout(2,j) = weightsin(2,j) + alpha * (t - z2) * input(j);
                end
                weightsout(2,3) = weightsin(2,3) + alpha * (t - z2);
            end
        end
    end
                   
                   
                   
                   
        