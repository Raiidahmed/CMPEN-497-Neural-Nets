%Raiid Ahmed Midterm Question 3

%Constructing the bipolar activation function

function [output] = bipolar_act(yin)

    if yin > 0 
        output = 1;
    else 
        output = -1;
    end
end