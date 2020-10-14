1;
function hough_lines_draw(img, outfile, peaks, rho, theta)
    % Draw lines found in an image using Hough transform.
    %
    % img: Image on top of which to draw lines
    % outfile: Output image filename to save plot as
    % peaks: Qx2 matrix containing row, column indices of the Q peaks found in accumulator
    % rho: Vector of rho values, in pixels
    % theta: Vector of theta values, in degrees
    f = figure
    imshow(img)
    % TODO: Your code here
    for i = 1:rows(peaks)
        thisRho = peaks(i, 1);
        thisTheta = peaks(i, 2);
        realRho = rho(thisRho);
        realTheta = theta(thisTheta);
        x = realRho * cos(realTheta); 
        y = - realRho * sin(realTheta); % -y because of other coord system
        t = 1000;
        v0 = y/x; 
        v1 = 1 ;
        p1 = [x y];
        p2 = [x y] + [v0 v1] * t;
        p3 = [x y] + [v0 v1] * (-t);
        % now our formula should be y = mx + c with two points
        %xlim = get(gca, 'Xlim');
        %leftY = m*xlim(1) + c
        %rightY = m*xlim(2) + c
        line([x p2(1) p3(1)], [y p2(2) p3(2)]);
    endfor
    saveas(f, outfile);
endfunction

% ps1
pkg load image;  % Octave only

%% 1-a
img = imread(fullfile('input', 'ps1-input0.png'));  % already grayscale
%% TODO: Compute edge image img_edges
img_edges = edge(img, 'canny');

imwrite(img_edges, fullfile('output', 'ps1-1-a-1.png'));  % save as output/ps1-1-a-1.png

%% 2-a
[H, theta, rho] = hough_lines_acc(img_edges, 'RhoResolution', 1, 'Theta', linspace(-90, 89, 180));  % defined in hough_lines_acc.m
%% TODO: Plot/show accumulator array H, save as output/ps1-2-a-1.png
%imshow(H, [])

imwrite(H, fullfile('output', 'ps1-2-a-1.png'))
%% 2-b
peaks = hough_peaks(H, 30);  % defined in hough_peaks.m
%% TODO: Highlight peak locations on accumulator array, save as output/ps1-2-b-1.png

%% TODO: Rest of your code here
%P  = houghpeaks(H,2);

hough_lines_draw(img, 'ps1-2-c-1.png', peaks, rho, theta);
