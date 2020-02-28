function [ intr, inte, labeltrain, labeltest, trpos2, tepos2 ] = TrainTestPixel( img, gt, trnum, cand )
%%
c = 1:max(max(gt));
intr = [];inte = [];labeltrain = [];labeltest = [];trpos = [];tepos = [];
[r1,c1,s1] = size(img);
img = reshape(img,r1*c1,s1);
gt = reshape(gt,r1*c1,1);
for i = 1:size(c,2 )
    pos = find(gt == i);
    index1 = randperm(size(pos,1));
    if(size(pos,1) <= trnum)
        trpos1 = pos(index1(1:cand));
        tepos1 = pos(index1(cand+1:length(index1)));
    else
        trpos1 = pos(index1(1:trnum));
        tepos1 = pos(index1(trnum+1:length(index1)));
    end
        intr = [intr;img(trpos1,:)];
        inte = [inte;img(tepos1,:)];
        labeltrain = [labeltrain;gt(trpos1,:)];
        labeltest = [labeltest;gt(tepos1,:)];
        trpos = [trpos;trpos1];
        tepos = [tepos;tepos1];
end

for i = 1:size(trpos,1)
    p1 = trpos(i,1);
   trpos2(i,:) = [mod(p1,r1),ceil(p1/r1)]; 
end
for i = 1:size(tepos,1)
    p1 = tepos(i,1);
   tepos2(i,:) = [mod(p1,r1),ceil(p1/r1)]; 
end
trpos2(find(trpos2 == 0)) = r1;
tepos2(find(tepos2 == 0)) = r1;

end

