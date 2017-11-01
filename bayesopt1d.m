%Bayesianopt
function x=bayesopt1d(f,range,epsilon)%NO NOISE
a=min(range);
b=max(range);
e=1;
xstar=(0:1/500:1)*(b-a)+a;
xstar=xstar';
k=unidrnd(501,3,1);
x=xstar(k);y=zeros(length(x),1);
for i=1:length(x)
y(i)=f(x(i));
end
hyp=[1;1];%hyperparameters
zstar=zeros(501,1);
for i=1:length(xstar)
zstar(i)=f(xstar(i));
end
fig=figure;
while e>epsilon
close(fig);
[ystar,K11,a]=gpinference(hyp,x,y,xstar);
y11=2*sqrt(diag(K11));
[~,ind]=max(a);
x0=xstar(ind);
e=min(max(y11),min(abs(x0-x)));
x=[x;x0];
y=[y;f(x0)];
fig=figure;
fill([xstar;flipud(xstar)], [ystar+y11; flipud(ystar-y11)],[.8 .8 .8],'Edgecolor','none');
hold on;
plot(xstar,zstar,'b','linewidth',3);
plot(xstar,ystar,'r','linewidth',3);
pause;
gif=figure;
plot(xstar,a,'g','linewidth',3);
pause;
close(gif);
disp(zstar-ystar);
end

end
function [ystar,K11,a]=gpinference(hyp,x,y,xstar)%EI function
a2=hyp(1);l=hyp(2);sigma=0;
k=@(x,y)(a2*exp(-(x-y)^2/(2*l^2)));
b=mean(y);y=y-b;
n=length(x);m=length(xstar);
K00=zeros(n,n);%K3=zeros(n,n);
K01=zeros(m,n);%K31=zeros(m,n);
K11=zeros(m,m);
%ystar=zeros(m,1);
for i=1:n
    for j=1:n
        K00(i,j)=k(x(i),x(j));
    end
    for j=1:m
        K01(j,i)=k(xstar(j),x(i));
    end
end
for i=1:m
    for j=1:m
        K11(i,j)=k(xstar(i),xstar(j));
    end
end
L=chol(K00+(sigma)*eye(n));
ystar=K01*solve_chol(L,y)+b;
K11=K11-K01*solve_chol(L,K01');
K11=(K11+K11.')/2;
z=sqrt(abs(diag(K11)));
fmax=max(y);
gamma=-(ystar-fmax)./z;
a=z.*(gamma.*normcdf(gamma)+normpdf(gamma));
%a=normcdf(gamma);
end
