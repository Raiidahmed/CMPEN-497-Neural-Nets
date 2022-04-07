%Raiid Ahmed Homework 5 Exercise 4.4

clc 
clear

w1old = [1;.8;.6;.4;.2];
x = [.5;1;.5;0;0;];
alpha = .2;

w1new = w1old + .2.*(x - w1old);