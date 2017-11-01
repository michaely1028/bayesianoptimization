%Bayesianopt
function [x,y,e,hyp]=bayesopt2dhyp(f,range,epsilon)%NO NOISE
%range=[x1,x2,y1,y2];

%===============initialization
a1=range(1);b1=range(2);
a2=range(3);b2=range(4);
e=1;
x1=(0:1/50:1)*(b1-a1)+a1;%------------------
x2=(0:1/50:1)*(b2-a2)+a2;
[m,n]=meshgrid(x1,x2);
xstar=[m(:) n(:)];
k=unidrnd(51^2,10,1);%----------
x=xstar(k,:);y=zeros(size(x,1),1);
for i=1:length(x)
y(i)=f(x(i,1),x(i,2));
end
%hyperparameters
zstar=zeros(51^2,1);
for i=1:length(zstar)
zstar(i)=f(xstar(i,1),xstar(i,2));
end
mesh(x1,x2,reshape(zstar,51,51));
pause;
fig=figure;

%=====================hyperparameter
hyp=[0;0;0];
fun=@(X)(negLogProb3(X,x,y));
[X, ~, i] = minimize(hyp, fun, 200);
hyp=exp(X);

%===========iteration
while e>epsilon
close(fig);
[ystar,K11,a]=gpinference(hyp,x,y,xstar);
y11=2*sqrt(diag(K11));
[~,ind]=max(a);
x0=xstar(ind,:);
test=sum(ones(length(x),1)*x0-x,2);
e=min(abs(test));
x=[x;x0];
y=[y;f(x0(1),x0(2))];
fig=figure;
%fill([xstar;flipud(xstar)], [ystar+y11; flipud(ystar-y11)],[.8 .8 .8],'Edgecolor','none');
mesh(x1,x2,reshape(ystar,51,51));
pause;
gif=figure;
mesh(x1,x2,reshape(a,51,51));
pause;
close(gif);
%disp(zstar-ystar);
end

end
function [ystar,K11,a]=gpinference(hyp,x,y,xstar)%EI function
a2=hyp(1);l1=hyp(2);l2=hyp(3);
%k=@(x1,x2,y1,y2)(a2*exp(-(x1-y1)^2/(2*l1^2)-(x2-y2)^2/(2*l2^2)));
k=@(x1,x2,y1,y2)(a2*exp(-(x1-y1)^2/(2*l1^2)-(x2-y2)^2/(2*l2^2)));
%k=@(x,y)(a2*exp(-(x-y)^2/(2*l^2)));
b=mean(y);y=y-b;
n=size(x,1);m=size(xstar,1);
K00=zeros(n,n);%K3=zeros(n,n);
K01=zeros(m,n);%K31=zeros(m,n);
K11=zeros(m,m);
%ystar=zeros(m,1);
for i=1:n
    for j=1:n
        K00(i,j)=k(x(i,1),x(i,2),x(j,1),x(j,2));
    end
    for j=1:m
        K01(j,i)=k(xstar(j,1),xstar(j,2),x(i,1),x(i,2));
    end
end
for i=1:m
    for j=1:m
        K11(i,j)=k(xstar(i,1),xstar(i,2),xstar(j,1),xstar(j,2));
    end
end
K00=(K00+K00.')/2;
L=chol(K00+(10^(-8))*eye(n));
ystar=K01*solve_chol(L,y)+b;
K11=K11-K01*solve_chol(L,K01');
K11=(K11+K11.')/2;
z=sqrt(abs(diag(K11)));
fmax=max(y);
gamma=-(ystar-fmax)./z;
a=z.*(gamma.*normcdf(gamma)+normpdf(gamma));
%a=normcdf(gamma);
end
