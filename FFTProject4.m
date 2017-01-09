%Adjust Parameters 
tmin = -0.005;
tmax = 0.005;
N=127;
dt = (tmax-tmin)/N;
Ns = 1;
%Assign Intensity
fs = (N-1)/(tmax-tmin);
t=linspace(tmin,tmax,N);
[tx, ty] = meshgrid(t);
It=zeros(size(tx));
for p = 1:Ns
It(randi(N),randi(N))= 1;
end
%FFT Things (Circshift rearranges the space so things transform
%appropriately)
Iu=circshift(circshift(fft2(circshift(circshift(It,(N+1)/2)',(N+1)/2)'),(N-1)/2)',(N-1)/2)';
[u,v] = meshgrid(fs/N*[-(N-1)/2:(N-1)/2]);
% Adjust Small  Parameters
ex = 0.5*dt;
ey = 0;
% Generate analytic solution
S = find(It);
tx0 = zeros(1,Ns);
ty0 = tx0;
Iue=zeros(size(It));
for p=1:Ns
tx0(1,p) = tx(S(p,1));
ty0(1,p) = ty(S(p,1));
Iue = Iue + exp(2*pi*1i*(u*(tx0(1,p)+ex)+v*(ty0(1,p)+ey)));
end
%Plot Things
figure(1)
surf(tx,ty,It)
title('I(theta)')
xlabel('tx')
ylabel('ty')
figure(2)
surf(u,v,real(Iu))
title('I(u)')
xlabel('u')
ylabel('v')
figure(3)
surf(u,v,real(Iue))
title('I(u+e)')
xlabel('u')
ylabel('v')
figure(4)
surf(u,v,real(Iu-Iue))
title('Difference')
xlabel('u')
ylabel('v')