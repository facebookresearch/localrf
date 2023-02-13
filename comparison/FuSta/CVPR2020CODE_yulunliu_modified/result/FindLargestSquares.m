function S = FindLargestSquares(I)
%FindLargestSquares - finds largest sqare regions with all points set to 1.
%input:  I - B/W boolean matrix
%output: S - for each pixel I(r,c) return size of the largest all-white square with its upper -left corner at I(r,c)  
[nr nc] = size(I);
S = double(I>0);
for r=(nr-1):-1:1
  for c=(nc-1):-1:1
    if (S(r,c))
      a = S(r  ,c+1);
      b = S(r+1,c  );
      d = S(r+1,c+1);
      S(r,c) = min([a b d]) + 1;
    end
  end  
end  