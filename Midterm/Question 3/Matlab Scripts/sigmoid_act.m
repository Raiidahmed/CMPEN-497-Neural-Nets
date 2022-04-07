%Raiid Ahmed Midterm Question 3

%Constructing the sigmoid activation function

function [output] = sigmoid_act(yin,sigma)
    
        output = (1./(1 + exp(-sigma .* yin)))* 2 - 1;
end