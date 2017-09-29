%This section defines initial parameters%
colors = ['y','m','c','r','g','b','k']' ;
Ns=1;
t=(-pi/2:pi/200:pi/2)';
lt=length(t);
I=zeros(lt,1);
tm=randperm(lt,Ns)';
for m=1:Ns
    I(tm(m,1),1)=1/randi(4,1)*200/pi;
end
figure(1)
plot(t,I)
title('I(theta)')
xlabel('theta (rad)')
[Iu,u]=FT1D(t,I,1);
Na=10;
lx=(lt+1)/2;
x=linspace(lx/lt*min(u),lx/lt*max(u),lx)';
xm=randperm(lx,Na)';
A=zeros(lx,1);
    for m = 1:Na
     A(xm(m,1),1) = 1/(x(5,1)-x(4,1));
    end
[y,B] = mycorrelate(x,A,A);
B(find(y==0),1) = 0;
figure(2)
plot(x,A)
title('Antennae')
xlabel('x (Principle Wavelength)')
%%
for p=0:6
    lambda = 0.94+0.02*p;
[Iu,u]=FT1D(t,I,lambda);
V=Iu.*B;
BIGV(:,p+1) = V;
[phi, PSF]=IFT1D(y,B,lambda);
BIGPSF(:,p+1) = PSF;
[theta,R]=IFT1D(u,V,lambda);
BIGR(:,p+1) = R;
figure(3)
hold on
plot(phi,real(PSF),strcat(colors(p+1,1),'--'))
title('PSF')
xlabel('phi (rad)')
figure(4)
hold on
plot(u,real(V),strcat(colors(p+1,1),'--'))
title('Visibility')
xlabel('u (wavelengths)')
figure(5)
hold on
plot(theta,real(R),strcat(colors(p+1,1),'--'))
title('Reconstructed Sky')
xlabel('theta (rad)')
end
BIGV = 1/7*sum(BIGV')';
BIGPSF = 1/7*sum(BIGPSF')';
BIGR = 1/7*sum(BIGR')';
figure(3)
hold on
plot(phi,real(BIGPSF),'LineWidth',5)
figure(4)
hold on
plot(u,real(BIGV),'LineWidth',5)
figure(5)
hold on
plot(theta,real(BIGR),'LineWidth',5)
