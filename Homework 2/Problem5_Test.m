output = [0 0 0 0 0     %Initial weights
          0 0 0 0 0];

alpha = .5;    %Set learning rate

for index = 1:4      %Train for 4 epochs, output not suppressed for debugging
    input = [1 1 1 1];
    weightsin = output;
    t = 1;

    [output, y, iserror] = Problem5_Function(input, weightsin, t, alpha)
    
    input = [-1 1 -1 -1];
    weightsin = output;
    t = 1;

    [output, y, iserror] = Problem5_Function(input, weightsin, t, alpha)
    
    input = [1 1 1 -1];
    weightsin = output;
    t = -1;

    [output, y, iserror] = Problem5_Function(input, weightsin, t, alpha)
    
    input = [1 -1 -1 1];
    weightsin = output;
    t = -1;

    [output, y, iserror] = Problem5_Function(input, weightsin, t, alpha)
end

input = [1,-1,-1,1];     %Test deployment

z1 = input*output(1,1:4).' + output(1,5)
z2 = input*output(2,1:4).' + output(2,5)