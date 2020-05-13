%% Michael Lustig CS exercise
%% 1). Sparse signals and Denoising
%% AUTHOR KAYLO LITTLEJOHN

%generate random 1x128 vector with 5 non-zero signals
clear all;
x = [[1:5]/5 zeros(1,128-5)];
x = x(randperm(128));

%add noise
y = x + 0.05*randn(1,128);

%estimate x_hat from y via Tychnovo penalty
gamma = [0.01 0.05 0.1 0.2];
figure("Name","Tychonov");
for i=1:4
    x_hat = (1/(1+gamma(i)))*y;
    subplot(2,2,i);
    plot(x_hat);
    title(strcat("gamma = ",num2str(gamma(i))));
end
sgtitle("Recovery via Tychonov Penalty");

%estimate x_hat from y via minimizing l^1 norm
figure("Name","l1 norm");
for i=1:4
    x_hat = SoftThresh(y,gamma(i));
    subplot(2,2,i);
    plot(x_hat);
    title(strcat("gamma = ",num2str(gamma(i))));
end
sgtitle("Recovery via L1 Norm Penalized Solution");
% RESULT -> L1 norm more sparse than Tychonov

%%
%compute fourier transform of sparse signal
X = fftc(x,2);

figure("Name","random");
subplot(2,2,1);
plot(x);
title("Random Signal x");

subplot(2,2,2);
plot(X);
title("Fourier Transform of x");

%equidistant undersampling
Xu = zeros(1,128);
Xu(1:4:128) = X(1:4:128);
xu = ifftc(Xu,2)*4;
subplot(2,2,3);
plot(abs(xu));
title("x' after Equidistant Undersampling");

%random undersampling
Xr = zeros(1,128);
prm = randperm(128);
Xr(prm(1:32)) = X(prm(1:32));
xr = ifftc(Xr,2)*4;
subplot(2,2,4);
plot(abs(xr))
title("x' after Random Undersampling");
% RESULT -> Random Undersampling results in turning ill-conditioned problem
% into a sparse signal denoising problem
sgtitle("Effects of sampling schema");



%%
%reconstruction from randomly undersampled f domain data 
%uses Projection Over Convex Sets type algorithm

%for at least 300 iterations ("projections")
Y = Xr;
X = Y;
nitr = 300;
error = zeros(1,nitr);
for i=1:nitr
    %compute ifft to get x_hat estimate from F_x_hat
    x_hat = ifftc(X,2);
    %apply Soft Thresh in the signal domain
    xst = SoftThreshComplex(x_hat,gamma(3));
    %compute fft of NEW x_hat xst
    X_curr = fftc(xst,2);
    %enforce data consistency in f domain
    X = X_curr.*(Y==0)+Y;
    %record error
    error(i) = norm(X-X_curr);
end

figure("Name","pocs results");

subplot(2,2,3)
plot(error)
title("Error");
ylabel("Normalized Error");
xlabel("Iterations");

subplot(2,2,2);
plot(abs(ifftc(X,2)));
title("Reconstructed Signal");

subplot(2,2,1);
plot(x);
title("Original Signal");
sgtitle("Reconstruction Results");

%% 2). Sparsity of Medical Images

%perform DWT and iDWT of phantom image
W = Wavelet; %generate wavelet operator
im = phantom(256); %phantom image
im_W = W*im; % get wavelet transform
im_rec = W'*im_W; % compute iDWT

%plot results
figure, subplot(1,3,1), imshow(im,[]), title('phantom image');
subplot(1,3,2), imshow(abs(im_W),[0,1]), title('The Wavelet coefficients');
subplot(1,3,3), imshow(abs(im_rec),[]), title('The reconstructed image');

%%
% load data set
brain = load('brain.mat');
im = brain.im;
im_W = W*im; % get wavelet transform

% plot wavelet coefficients
figure("Name","wavelet coef");
subplot(2,2,1);
imagesc(abs(im));
colormap("gray");
title("Original Image");
subplot(2,2,2);
imshowWAV(im_W);
title("Wavelet Coeffecients");

% threshold largest % of coefficients
faction = 0.05; % percent of coeffecients to threshold
m = sort(abs(im_W(:)),'descend');
ndx = floor(length(m)*faction);
thresh = m(ndx);
im_W_th = im_W .* (abs(im_W)>thresh);
subplot(2,2,3);
imshowWAV(im_W_th);
title("Thresholded Wavelet Coeffecients");
subplot(2,2,4);
imagesc(abs(im) - abs(W'*im_W_th));
title("Difference Image of Reconstruction");
figure("Name","recon");
imagesc(abs(W'*im_W_th));
title("Reconstructed Image");
colormap("gray");

%% 3). Compressed Sensing Reconstruction

%load image into matlab
brain = load('brain.mat');
im = brain.im;
im_W = W*im; % get wavelet transform

%get undersampling patterns
mask_unif = brain.mask_unif;
pdf_unif = brain.pdf_unif;
mask_vardens = brain.mask_vardens;
pdf_vardens = brain.pdf_vardens;

M = fft2c(im); %compute fft of image
Mu = (M.*mask_unif)./pdf_unif; %multiply by mask, divide by PDF
imu = ifft2c(Mu);
Mnu = (M.*mask_vardens)./pdf_vardens;
imnu = ifft2c(Mnu);

%plot results
figure("Name","CS_uni");
subplot(2,2,1);
imagesc(abs(im));
title("Original Image");
subplot(2,2,2);
imagesc(abs(imu));
title("Masked Image");
subplot(2,2,3);
imagesc(abs(im-imu));
title("Difference Image");
colormap("Gray");
sgtitle("Effects of Uniform Random Sampling");

figure("Name","CS_nuni");
subplot(2,2,1);
imagesc(abs(im));
title("Original Image");
subplot(2,2,2);
imagesc(abs(imnu));
title("Masked Image");
subplot(2,2,3);
imagesc(abs(im-imnu));
title("Difference Image");
colormap("Gray");
sgtitle("Effects of Non-Uniform Random Sampling");

%%
%implement POCS algorithm as done in P1.
Y = Mnu;
X = Y;
gam = 0.2;
nitr = 20;
error = zeros(1,nitr);
for i=1:nitr
    %compute ifft to get x_hat estimate from F_x_hat
    x_hat = ifft2c(X);
    %apply dwt
    x_hat_W = W*x_hat;
    %apply Soft Thresh in the signal domain
    xst_W = SoftThreshComplex2D(x_hat_W,gam);
    %apply idwt
    xst = W'*xst_W;
    %compute fft of NEW x_hat xst
    X_curr = fft2c(xst);
    %enforce data consistency in f domain
    X = X_curr.*(Y==0)+Y;
    %record error
    error(i) = norm(X-X_curr);
end

% plot results
figure("Name","pocs results");
subplot(2,2,3)
plot(error)
title("Error");
ylabel("Normalized Error");
xlabel("Iterations");

subplot(2,2,2);
imagesc(abs(ifft2c(X)));
title("Reconstructed Signal");

subplot(2,2,1);
imagesc(abs(im));
title("Original Signal");
colormap("gray");
sgtitle("Reconstruction Results");

%accepts y and gamma, returns x_hat
function x_hat = SoftThresh(y,gamma)
    x_hat = zeros(size(y));
    for i=1:length(y)
        if(y(i)<-gamma)
            x_hat(i)=y(i)+gamma;
        end
        if(y(i)>gamma)
            x_hat(i)=y(i)-gamma;
        end
    end
end

%accepts y and gamma, returns x_hat
function x_hat = SoftThreshComplex(y,gamma)
    x_hat = zeros(size(y));
    for i=1:length(y)
        if(abs(y(i))>gamma)
            x_hat(i)=((abs(y(i))-gamma)/abs(y(i)))*y(i);
        end
    end
end

%accepts y and gamma, returns x_hat
function x_hat = SoftThreshComplex2D(y,gamma)
    x_hat = zeros(size(y));
    [M, N] = size(y);
    for i=1:M
        for j = 1:N
            if(abs(y(i,j))>gamma)
                x_hat(i,j)=((abs(y(i,j))-gamma)/abs(y(i,j)))*y(i,j);
            end
        end
    end
end

function res = fftc(x,dim)
% res = fftc(x,dim)
res = 1/sqrt(size(x,dim))*fftshift(fft(ifftshift(x,dim),[],dim),dim);
end

function res = ifftc(x,dim)
%res = ifftc(x,dim)
res = sqrt(size(x,dim))*fftshift(ifft(ifftshift(x,dim),[],dim),dim);
end

function imshowWAV(w)
	imshow(abs(wavMask(size(w),1).*w),[]);
end

function res = wavMask(imSize,scale)
% function scales the value of each wavelet scale such they display nicely.

sx = imSize(1);
sy = imSize(2);
res = zeros(imSize)+1;
NM = round((log2(imSize)));
for n = 1:min(NM)-scale+1
	res(1:round(2^(NM(1)-n)),1:round(2^(NM(2)-n))) = res(1:round(2^(NM(1)-n)),1:round(2^(NM(2)-n)))/2;
end
end

function res = fft2c(x)

% res = fft2c(x)
% 
% orthonormal forward 2D FFT
%
% (c) Michael Lustig 2005

res = 1/sqrt(length(x(:)))*fftshift(fft2(ifftshift(x)));
end

function res = ifft2c(x)
%
%
% res = ifft2c(x)
% 
% orthonormal centered 2D ifft
%
% (c) Michael Lustig 2005

res = sqrt(length(x(:)))*ifftshift(ifft2(fftshift(x)));
end
