%***************************
%*  Gradient With Correct Edges
%*  Mahmoud Farshbaf Doustar
%* 
%*  2010,3,10
%*
%* Description:
%* Making a Canvas one pixel larger than the image in each direction
%* Filling Canvas with 1
%* Compute Gradients in X and Y, 
%* This time edges have value and don''t make artifact in furthure HOG
%* processing
%***************************
%
%  I : input Image
%  Fx: Gradient in X direction
%  Fy: Gradient in Y direction
% 
% 
%******************************

function [Fx,Fy]=myGradient(I)
%correctig gradient on edges
I=imadjust(I);
[hcol,hrow]=size(I);
temp= ones(hcol+2,hrow+2);
temp(2:1+hcol,2:1+hrow)=I(:,:);

Fx=filter2([1;0;-1],temp);
Fx=Fx(3:hcol,3:hrow);
Fy=filter2([1;0;-1]',temp);
Fy=Fy(3:hcol,3:hrow);

