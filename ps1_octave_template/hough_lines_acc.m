1;
function [H, theta, rho] = hough_lines_acc(BW, varargin)
    % Compute Hough accumulator array for finding lines.
    %
    % BW: Binary (black and white) image containing edge pixels
    % RhoResolution (optional): Difference between successive rho values, in pixels
    % Theta (optional): Vector of theta values to use, in degrees
    %
    % Please see the Matlab documentation for hough():
    % http://www.mathworks.com/help/images/ref/hough.html
    % Your code should imitate the Matlab implementation.
    %
    % Pay close attention to the coordinate system specified in the assignment.
    % Note: Rows of H should correspond to values of rho, columns those of theta.

    %% Parse input arguments
    p = inputParser();
    addParameter(p, 'RhoResolution', 1);
    addParameter(p, 'Theta', linspace(-90, 89, 180));
    %p = p.addParameter('RhoResolution', 1);
    %p = p.addParameter('Theta', linspace(-90, 89, 180));
    parse(p, varargin{:});
    rhoStep = p.Results.RhoResolution;
    theta = p.Results.Theta;
    rhoMax = ceil(sqrt(columns(BW) ** 2 + rows(BW) ** 2));

    disp(rhoMax)
    rho = linspace(-rhoMax, rhoMax, ceil(rhoMax / rhoStep) * 2 + 1);
    H = zeros(ceil(rhoMax / rhoStep) * 2, numel(theta));
    disp(columns(H));
    disp(rows(H));
    for x = 1:columns(BW)
        for y = 1:rows(BW)
            if BW(x, y) == 1
                for thisThetaIdx = 1:numel(theta);
                    thisTheta = theta(thisThetaIdx);
                    thisRho = x * cos(thisTheta) - y * sin(thisTheta);
                    [minVal, thisRhoIdx] = min(abs(rho - thisRho));
                    %thisRhoIdx = round((thisRho - rho(1)) / rhoStep) + 1;
                    H(thisRhoIdx, thisThetaIdx) += 1;
                endfor
            endif
        endfor
    endfor
    #surf(H)
    #imshow(H, [])
    min(H(:))
    max(H(:))
    %% TODO: Your code here
endfunction

img = imread(fullfile('input', 'ps1-input0.png'));  % already grayscale
%% TODO: Compute edge image img_edges
img_edges = edge(img, 'canny');

%% 2-a
[H, theta, rho] = hough_lines_acc(img_edges, 'RhoResolution', 1, 'Theta', linspace(-90, 89, 180));