%Raiid Ahmed Midterm Question 3

%Constructing the sigmoid derivative for extended delta rule

function [output] = sigmoiddiff_act(yin,sigma)
    
      output = (sigma/2)*(1 + sigmoid_act(yin,sigma))*(1 - sigmoid_act(yin,sigma));
end