clc;
clear;

input='result.avi';
mask_file='result.png';
output='result_cropped.avi';


mask=imread(mask_file);
mask=double(mask>0);


[C, H, W, M] = FindLargestRectangles(mask,[0,0,1]);
[minx,miny]=ind2sub(size(M),min(find(M)));
[maxx,maxy]=ind2sub(size(M),max(find(M)));

v=VideoReader(input);
vo=VideoWriter(output);
vo.FrameRate=v.FrameRate;
open(vo);
for j=1:v.NumberOfFrames
    frame=read(v,j);
    frame=frame(minx:maxx,miny:maxy,:);
    writeVideo(vo,frame);
end
close(vo);
