%Raiid Ahmed Question 2
%Just some operations to help with calculating weights

clc
clear

x1 = [1;
      1;
      1;
      1;
      1];

x2 = [1;
      -1;
      -1;
      1;
      -1];

x3 = [-1;
       1;
      -1;
       1;
       1];
   
w1 = x1 * x1.';
w2 = x2 * x2.';
w3 = x3 * x3.';

w = w1 + w2 + w3;
w = w - diag(diag(w));

order = randperm(5);
       