1;
function peaks = hough_peaks(H, varargin)
    % Find peaks in a Hough accumulator array.
    %
    % Threshold (optional): Threshold at which values of H are considered to be peaks
    % NHoodSize (optional): Size of the suppression neighborhood, [M N]
    %
    % Please see the Matlab documentation for houghpeaks():
    % http://www.mathworks.com/help/images/ref/houghpeaks.html
    % Your code should imitate the matlab implementation.

    %% Parse input arguments
    p = inputParser;
    addOptional(p, 'numpeaks', 1, @isnumeric);
    addParameter(p, 'Threshold', 0.5 * max(H(:)));
    addParameter(p, 'NHoodSize', floor(size(H) / 100.0) * 2 + 1);  % odd values >= size(H)/50
    parse(p, varargin{:});

    numpeaks = p.Results.numpeaks;
    threshold = p.Results.Threshold;
    nHoodSize = p.Results.NHoodSize;
    Q = [];
    %disp(nHoodSize)
    for i = 1:numpeaks
        [maxVal, maxLinIndex] = max(H(:));
        [row col] = find(H == maxVal);
        thisRow = row(1);
        thisCol = col(1);
        Q = [Q; [thisRow thisCol]];
        rowDel = floor(nHoodSize(1) / 2);
        colDel = floor(nHoodSize(2) / 2);
        H(row-rowDel:row+rowDel, col-colDel:col+colDel) = 0;
    endfor
    peaks = Q;
    % TODO: Your code here
endfunction

% ps1
pkg load image;  % Octave only

%% 1-a
img = imread(fullfile('input', 'ps1-input0.png'));  % already grayscale
%% TODO: Compute edge image img_edges
img_edges = edge(img, 'canny');

imwrite(img_edges, fullfile('output', 'ps1-1-a-1.png'));  % save as output/ps1-1-a-1.png

%% 2-a
[H, theta, rho] = hough_lines_acc(img_edges, 'RhoResolution', 5);  % defined in hough_lines_acc.m
%% TODO: Plot/show accumulator array H, save as output/ps1-2-a-1.png
%imshow(H, [])

imwrite(H, fullfile('output', 'ps1-2-a-1.png'))
%% 2-b
peaks = hough_peaks(H, 10)  % defined in hough_peaks.m
%% TODO: Highlight peak locations on accumulator array, save as output/ps1-2-b-1.png

%% TODO: Rest of your code here
