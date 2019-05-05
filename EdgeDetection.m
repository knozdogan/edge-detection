classdef EdgeDetection
    properties
        img
        kernelSize
        sigma
        threshold
        strong
        weak
    end
    properties(Constant)
        wk = 30
    end
    
    methods
        function [grad, dir] = Sobel(~,img)
            SobelKernelX = [-1 0 1;-2 0 2;-1 0 1];
            SobelKernelY = [1 2 1;0 0 0;-1 -2 -1];
            gX = conv2(img, SobelKernelX, 'same');
            gY = conv2(img, SobelKernelY, 'same');
            grad = sqrt(gX.^2+gY.^2);
            grad = uint8(grad);
            dir = atan2(gX, gY);
        end
        
        function result = SobelDetection(obj)
            [Grad, ~] = obj.Sobel(obj.img);
            Grad(Grad<obj.threshold) = 0;
            Grad(Grad>obj.threshold) = 255;
            result = obj.convertImg(Grad);
        end
        
        function result_filter = GaussianFilter(obj)
            normalization = 1/(2*pi*(obj.sigma^2));
            denom = 2*obj.sigma^2;
            kernel = zeros(obj.kernelSize, obj.kernelSize);
            k = (obj.kernelSize-1)/2;
            for i=1:obj.kernelSize
                for j=1:obj.kernelSize
                    kernel(i,j) = normalization*exp(-((i-k-1)^2+(j-k-1)^2)/denom);
                end
            end
            result_filter = conv2(obj.img, kernel, 'same');
        end
        
        function result = basicEdgeDetection(obj)
            im = padarray(obj.img, [1 1], 'both');
            [m,n] = size(im);
            im = double(im);
            grad = zeros(m-2,n-2);
            for i=1:m-2
                for j=1:n-2
                    grad(i,j)=sqrt((im(i+1,j+1)-im(i+1,j))^2+(im(i+1,j+1)-im(i,j+1))^2);
                end
            end
            grad = uint8(grad);
            grad(grad<obj.threshold) = 0;
            grad(grad>obj.threshold) = 255;
            result = obj.convertImg(grad);
        end
        
        function result = Canny(obj)
            direction = [0 45 90 135 180 225 270 315 360];
            filter_out = obj.GaussianFilter;
            [grad, dir] = obj.Sobel(filter_out);
            im = padarray(filter_out,[1 1],'both');
            degree = rad2deg(dir);
            degree = wrapTo360(degree);
            [k,l] = size(grad);
            result = zeros(k,l);
            for i=1:k
                for j=1:l
                    abs_ = abs(direction-degree(i,j));
                    val = min(abs_);
                    idx = find(abs_==val);
                    if idx==1 || idx==5 || idx==9
                        right = im(i+1,j+2);
                        left = im(i+1,j);
                    elseif idx==2 || idx==6
                        right = im(i,j+2);
                        left = im(i+2,j);
                    elseif idx==3 || idx==7
                        right = im(i,j+1);
                        left = im(i+2,j+1);
                    elseif idx==4 || idx==8
                        right = im(i,j);
                        left = im(i+2,j+2);
                    end
                    if im(i+1,j+1)>=right && im(i+1,j+1)>=left
                        result(i,j)=im(i+1,j+1);
                    end
                end
            end
            result = obj.normalize(result);
            result = obj.doubleThreshold(result);
            result = obj.convertImg(result);
            %result = obj.hysteresis(result);
%           
%             out = obj.hysteresis(dTh);
        end
        
        function out = doubleThreshold(obj,input)
            [k,l] = size(input);
            out = zeros(k,l);
            for i=1:k
                for j=1:l
                    if input(i,j)>=obj.strong
                        out(i,j)=255;
                    elseif input(i,j)<obj.strong && input(i,j)>=obj.weak
                        out(i,j)=obj.wk;
                    end
                end
            end
        end
        
        function out = hysteresis(obj,input)
            [k,l] = size(input);
            out = zeros(k,l);
            for i=2:k-1
                for j=2:l-1
                    if input(i,j)== obj.wk
                        if input(i+1,j)==255 || input(i+1,j+1)==255 || input(i,j+1)==255 || ...
                           input(i-1,j+1)==255 || input(i-1,j)==255 || input(i-1,j-1)==255 || ...
                           input(i,j-1)==255 || input(i+1,j-1)==255
                            input(i,j)=255;
                        end
                    end
                end
            end
        end
        
        function result = convertImg(~, input)
            [m,n] = size(input);
            result = zeros(m,n);
            for i=1:m
                for j=1:n
                    if input(i,j)==0
                        result(i,j)=255;
                    else
                        result(i,j)=0;
                    end
                end
            end
        end
        
        function normalized = normalize(~,img)
            d=linspace(min(img(:)), max(img(:)),256);
            normalized = uint8(arrayfun(@(x) find(abs(d(:)-x)==min(abs(d(:)-x))), img));
        end
    end
end