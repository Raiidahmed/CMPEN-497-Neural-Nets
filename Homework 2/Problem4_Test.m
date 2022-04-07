
output = [.01 .1 .15     %Initial weights
          .02 .1  .3];

alpha = .5;     %Set learning rate

for index = 1:4       %Train for 4 epochs, output not suppressed for debugging
    input = [1,1];
    weightsin = output;
    t = -1;

    [output, y, iserror] = Problem4_Function(input, weightsin, t, alpha)

    input = [1,-1];
    weightsin = output;
    t = 1;

    [output, y, iserror] = Problem4_Function(input, weightsin, t, alpha)

    input = [-1,1];
    weightsin = output;
    t = 1;

    [output, y, iserror] = Problem4_Function(input, weightsin, t, alpha)

    input = [-1,-1];
    weightsin = output;
    t = -1;

    [output, y, iserror] = Problem4_Function(input, weightsin, t, alpha)
end

input = [-1,-1];        %Test deployment

z1 = input*output(1,1:2).' + output(1,3)
z2 = input*output(2,1:2).' + output(2,3)


