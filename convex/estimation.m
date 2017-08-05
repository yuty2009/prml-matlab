clc
clear
%%%%%%%%%%%%%%
%creat data
miu1=1;miu2=10;
sigma1=1; sigma2=1;
x1=randn(1000,1)*sigma1+miu1;
x2=randn(3000,1)*sigma2+miu2;
x=[x1;x2];
% hist(x);
for i=1:4000
 x(i)=x(i,1);
end
%%%%%%%%%%%%%%%%%%%%%%%%%
%include wi in v1(i)
n=1;n1=1;
miu1(n)=1.2;miu2(n)=10.5;%provide an initial data for mean and var
sigma1=1; sigma2=1;
for i=1:4000
if i<=1000
    miu=miu1(n);sigma=sigma1;
    v1(i)=log((1/4)*(((2*pi)^n1*det(sigma))^-0.5)) -(x(i)-miu)'*(inv(sigma))*(x(i)-miu)/2;
    c1(i)=log((1/4)*(((2*pi)^n1*det(sigma))^-0.5));
    A1{i,1}=[x(i);-1;0]*[x(i);-1;0]'*(1/2);
else 
   miu=miu2(n);sigma=sigma2;
   v1(i)=log((3/4)*(((2*pi)^n1*det(sigma))^-0.5)) -(x(i)-miu)'*(inv(sigma))*(x(i)-miu)/2;
   c1(i)=log((3/4)*(((2*pi)^n1*det(sigma))^-0.5));
   A1{i,1}=[x(i);0;-1]*[x(i);0;-1]'*(1/2);
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%
%include wi in v2(i)
for i=1:4000
if i<=1000
    miu=miu2(n);sigma=sigma2;
    v2(i)=log((3/4)*(((2*pi)^n1*det(sigma))^-0.5)) -(x(i)-miu)'*(inv(sigma))*(x(i)-miu)/2;
    c2(i)=log((3/4)*(((2*pi)^n1*det(sigma))^-0.5));
    A2{i,1}=[x(i);0;-1]*[x(i);0;-1]'*(1/2);
else 
   miu=miu1(n);sigma=sigma1;
   v2(i)=log((1/4)*(((2*pi)^n1*det(sigma))^-0.5)) -(x(i)-miu)'*(inv(sigma))*(x(i)-miu)/2;
   c2(i)=log((1/4)*(((2*pi)^n1*det(sigma))^-0.5));
   A2{i,1}=[x(i);-1;0]*[x(i);-1;0]'*(1/2);
end
end
%%%%%%%%%%%%%%%%%%%%%%
%define A,Z,c,Q,r
r=2;
ZZ=[1,miu1(n),miu2(n)]'*[1,miu1(n),miu2(n)];
Q{n,1}=[miu1(n);-1;0]*[miu1(n);-1;0]'+[miu2(n);0;-1]*[miu2(n);0;-1]';
for i=1:4000
    c(i)=c1(i)-c2(i);
    A{i,1}=A1{i,1}-A2{i,1};
  % Y=[miu1(n),miu2(n)]'*[miu1(n),miu2(n)];
end
%%%%%%%%%%%%%%%%%%%%%%
%convex optimization
%c0(n)=0.01;
n=n+1;
m= size(ZZ,1);
% c0=0.01;
cvx_begin
   variable Z(m,m) symmetric;
   variable c0;
   minimize(-c0);
   subject to
   for i=1:4000
    trace(A{i,1}*Z)+c0<=c(i);
   end
   trace(Q{n-1,1}*Z)<=r^2;
   Z(1,1)==1;
   Z==semidefinite(m);
   c0>=0;
cvx_end
%%%%%%%%%%%%%%%%%%%%%%%%%
%prove v1-v2 equal to c(i)-trace(A{i,1}*Z)
% for i=1:4000
%     l(i,1)=c(i)-trace(A{i,1}*Z);
% end
% l1=v1-v2;
%%%%%%%%%%%%%%%%%%%%%%%%%%
%define logGaussian
%function v=logGaussian(x, mean, var)
%n = size(x, 1);
%v = -0.5 * log((2*pi)^n*det(var)) -(x-mean)'*inv(var)*(x-mean)/2;