% ps1
pkg load image;  % Octave only

%% 1-a
img = imread(fullfile('input', 'ps1-input0.png'));  % already grayscale
%% TODO: Compute edge image img_edges
img_edges = edge(img, 'canny');

imwrite(img_edges, fullfile('output', 'ps1-1-a-1.png'));  % save as output/ps1-1-a-1.png

%% 2-a
[H, theta, rho] = hough_lines_acc(img_edges);%, 'RhoResolution', 5);  % defined in hough_lines_acc.m
%% TODO: Plot/show accumulator array H, save as output/ps1-2-a-1.png
%imshow(H, [])
disp(theta)
disp(rho)
imwrite(H, fullfile('output', 'ps1-2-a-1.png'))
%% 2-b
peaks = hough_peaks(H, 10);  % defined in hough_peaks.m
%% TODO: Highlight peak locations on accumulator array, save as output/ps1-2-b-1.png

%% TODO: Rest of your code here
%P  = houghpeaks(H,2);
imshow(H,[],'XData',theta,'YData',rho,'InitialMagnification','fit');
xlabel('\theta'), ylabel('\rho');
axis on, axis normal, hold on;
plot(theta(peaks(:,2)),rho(peaks(:,1)),'s','color','green');

hough_lines_draw(img, 'ps1-2-c-1.png', peaks, rho, theta);

% for comparison the matlab code
[H, t, r] = hough(img_edges);
peaks2 = houghpeaks(H, 10);
disp(peaks)
disp(" ")
disp(peaks2)
imshow(H);
%hough_lines_draw(img, 'ps1-2-c-2.png', peaks2, r, t);
t = t
r = r