%%
% If you want to conduct experiments on other datasets, you should change
% the dataset name (namely "IP") here. Besides, the variables "data_name" and
% "num_classes" in the file "trainMDGCN.py" should also be modified
% accordingly.
%%
% 'train_num' denotes the number of labeled pixels in each land cover class
% statistical results are saved in the file 'stat_res.mat'
% the predicted labels are saved in the file 'pred_mat.mat'
%%
train_num = 30;
load('./data/IP_gyh');
load('./data/IP_gt');
[ ~, ~, ~, ~, trpos,tepos ] = TrainTestPixel(IP_gyh, IP_gt, train_num, 15 );
save('./data/trpos', 'trpos');
save('./data/tepos', 'tepos');
% You may need to add the path of python manually
system('python trainMDGCN.py');
