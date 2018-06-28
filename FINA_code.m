clc;
clear all;
close all
 warning off all
 addpath cd 
 %%
k = dir('data') ;  % loading  files in the directory data 
 count_total=1;
 first=1;
 last=50;
  label_im=1;
  %% ------------CHANGE --------------------- 'F:\Payement receipt\
  dataDir = fullfile('D:\mridul_finalcode\data set');
ImgCovers = imageSet(dataDir);
imageIndex = image_feature_initialize(ImgCovers,'Verbose',false); % removing verbos will make code slaower and detailed

%  features=zeros((length(k)-4),20);
 final_feat =zeros(18,50);  
 final_label=zeros(18,50); % 5 images and 50 features  of zeros 
 for i=3:length(k) % finding length of it  
 files=dir(strcat('data\',k(i).name)) ;  % loading files fom on efolder at a time
  inside_count=1;
     for j=3:length(files) 
      
    path=(k(i).name);
    image =files(j).name ;
    full_path=strcat(['data\',path,'\',image]);
    img=imread(full_path);
     %%
     
Igray = imresize(img,[400 400]);
  
Ibw = im2bw(Igray,graythresh(Igray));
 
Iedge = edge(uint8(Ibw));
  
 corners = detectHarrisFeatures(Iedge);
  
 
features_cornerpoints=corners.selectStrongest(50);
se = strel('square',3);
Iedge2 = imdilate(Iedge, se);
  
Ifill= imfill(Iedge2,'holes');
 
[Ilabel ,num] = bwlabel(Ifill);
Iprops = regionprops(Ilabel);
Ibox = [Iprops.BoundingBox];
[y,x]=size(Ibox);
x=x/4;
Ibox = reshape(Ibox,[4 x]);
Ic = [Iprops.Centroid];
[z,w]=size(Ic);%
w=w/2;%
Ic = reshape(Ic,[2 w]);
Ic = Ic';
Ic(:,3) = (mean(Ic.^2,2)).^(1/2);
Ic(:,4) = 1:w;
Ic2 = sortrows(Ic,2);
Ic2_len=length(Ic2(:)');
if Ic2_len < 29
  add_zero= 28-Ic2_len;
  ab=zeros(1,add_zero);
end 
 Ic2=Ic2(:)';
 x_loc =features_cornerpoints.Location';
 feat_1(inside_count,:)=x_loc(1,:);
 feat_2(inside_count,:)=x_loc(2,:);
 feat_3(inside_count,:)=features_cornerpoints.Metric;

 feat=[feat_1; feat_2 ;feat_3];
 
%  value=features_cornerpoints.Metric';
  
     count_total=count_total+1;
    inside_count=inside_count+1;
%     label(label_im,first:last)=1;
                     
     end  
%       label_im=label_im+1
%      first=last+1
%      last=first+50
[m,n]=size(feat);
 
      final_feat=[final_feat feat];  
       
      
      
%        a=[ones(count,count_total),zeros(1, length(k)-count_total)]; 
%      count=count+1;
%      label(count,first:last)=ones 
     
 end 
   final_feat(:,1:50)=[];
   
   %%
   first=1;
   last=50;
   total_set=length(k)-2;
   labels=zeros( total_set, total_set*50);
   tp=1;
   for kp=1:total_set
   labels(tp,first:last)=1;
   first=last+1;
   last=last+50;
    tp=tp+1;
   end 
%  [Best_features,weights] = MI(final_feat, labels);
%  %%
%  groups=features(:,end);
%  feature_train=features(:,end-1);% [features(:,1) features(:,2) features(:,3) features(:,4) features(:,8) features(:,7) features(:,6) features(:,5) features(:,10)];
%  %%
%  label=zeros( 12,72); 
%  ct=1;
%  ctk=6;
%  for i=1:12
%  label(i,ct:ctk)=1 ; 
%   ct=ctk+1;
%   ctk=ctk+6;
%  
%  end 
%  ctp=ct;
%  %%
% %  label_b=ones(1,count_b-1);
% % label_m=ones(1,count_m-1);
% % total=(count_m-1)+count_b-1;
% % k=zeros((total-(count_b-1)),1)';
% % label_benign=[label_b  k ];
% % 
% %  label_malignant=1-label_benign;
% % 
  neur_label=labels;
%  
  neural_feat= final_feat;
% %
  net = patternnet(50);
  net.performParam.regularization = 0.1;

 [net1,tr1] = train(net,neural_feat,neur_label);
%    

y = net1(neural_feat);
perf=crossentropy(net1,neur_label,y,{1},'regularization',0.1)
  nntraintool 
  plotperform(tr1)
  testX1 = neural_feat(:,tr1.testInd);
  testT1 = neur_label(:,tr1.testInd);
  testY1 = net1(testX1);
  testIndices = vec2ind(testY1);
    figure,plotconfusion(testT1,testY1)
   pause(0.002)
     [c1,cm1] = confusion(testT1,testY1);
 %% TESTING PART 
 % Select and display the query image.
 [filename,pathname]=uigetfile('*.png*','LOAD TEST FILE');
 
queryImage = imread(strcat(pathname,filename));%'NF4.png');

figure
imshow(queryImage)

%% 
  [imageIDs, ~, queryWords] = retrieveImages(queryImage, imageIndex);

%% 
% Find the best match for the query image by extracting the visual words from the image index. The image index contains the visual word information for all images in the index.
bestMatch = imageIDs(1);
bestImage = imread(imageIndex.ImageLocation{bestMatch}); % LOCATION OF MATCHING SIGNATURE 
bestMatchWords = imageIndex.ImageWords(bestMatch);

%% 
% Generate a set of tentative matches based on visual word assignments. Each visual word in the query can have multiple matches due to the hard quantization used to assign visual words.
queryWordsIndex     = queryWords.WordIndex;
bestMatchWordIndex  = bestMatchWords.WordIndex;

tentativeMatches = [];
for i = 1:numel(queryWords.WordIndex)
    
    idx = find(queryWordsIndex(i) == bestMatchWordIndex);
    
    matches = [repmat(i, numel(idx), 1) idx];
    
    tentativeMatches = [tentativeMatches; matches];
    
end

%% 
% Show the point locations for the tentative matches. There are many poor matches.
points1 = queryWords.Location(tentativeMatches(:,1),:);
points2 = bestMatchWords.Location(tentativeMatches(:,2),:);
 

figure
showMatchedFeatures(queryImage,bestImage,points1,points2,'montage')

%% 
% Remove poor visual word assignments using |estimateGeometricTransform| function. Keep the assignments that fit a valid geometric transform.
[tform,inlierPoints1,inlierPoints2] = estimateGeometricTransform(points1,points2,'affine');

%% 
% Rerank the search results by the percentage of inliers. Do this when the geometric verificiation procedure is applied to the top _N_ search results. Those images with a higher percetage of inliers are more likely to be relevant.
percentageOfInliers = size(inlierPoints1,1)./size(points1,1);
 

figure
showMatchedFeatures(queryImage,bestImage,inlierPoints1,inlierPoints2,'montage')
%% 
% Apply the estimated transform.
outputView = imref2d(size(bestImage));
Ir = imwarp(queryImage, tform, 'OutputView', outputView);


figure
imshowpair(Ir,bestImage,'montage')

 %%
 
img_original=imread(imageIndex.ImageLocation{bestMatch});
 
[file,path,ext]=fileparts(imageIndex.ImageLocation{bestMatch});
output=path(end-2:end);
%% CHECK QUERY 
best= dir(strcat('D:\mridul_finalcode\GENUINE\',output));
 %%
ctn=1;
%      % for original image
for i=3:length(best)
    stp=imread(strcat('GENUINE','\',output,'\',best(i).name));
 Igray_original = imresize(stp,[400 400]);
 Ibw_original = im2bw(Igray_original,graythresh(Igray_original));
 Iedge_original = edge(uint8(Ibw_original));
 se = strel('square',3);
 Iedge_original = imdilate(Iedge_original, se);
corners = detectHarrisFeatures(Iedge);
features_cornerpoints=corners.selectStrongest(50);
%  % For test image 
  I2_test=queryImage; % ----------test image should come here 
%  
Igray_test = imresize(I2_test,[400 400]);
Ibw_test = im2bw(Igray_test,graythresh(Igray_test));
 Iedge_test = edge(uint8(Ibw_test));
 se = strel('square',3);
 Iedge_test = imdilate(Iedge_test, se); 
%  % for harris features 
  
 points1 = detectHarrisFeatures(Iedge_original);
 points2 = detectHarrisFeatures(Iedge_test);
 [features1,valid_points1] = extractFeatures(Iedge_original,points1);
 [features2,valid_points2] = extractFeatures(Iedge_test,points2);
 indexPairs = matchFeatures(features1,features2);
 matchedPoints1 = valid_points1(indexPairs(:,1),:);
 matchedPoints2 = valid_points2(indexPairs(:,2),:);
  %figure; showMatchedFeatures(Iedge_original,Iedge_test,matchedPoints1,matchedPoints2); 
 
% % for Surf 
%  
 points1 = detectSURFFeatures(Iedge_original);
points2 = detectSURFFeatures(Iedge_test);
 
[f1,vpts1] = extractFeatures(Iedge_original,points1);
[f2,vpts2] = extractFeatures(Iedge_test,points2);
 
indexPairs = matchFeatures(f1,f2) ;
matchedPoints1 = vpts1(indexPairs(:,1));
matchedPoints2 = vpts2(indexPairs(:,2));
 
 figure; showMatchedFeatures(Iedge_original,Iedge_test,matchedPoints1,matchedPoints2);
legend('matched points 1','matched points 2');
%  
%  %
%   
 data1=matchedPoints1.Metric;
data2=matchedPoints2.Metric;
% 
% 
% 
% %applying edge detection on first picture
% %so that we obtain white and black points and edges of the objects present
% %in the picture.
% 
edge_det_pic1 = edge(Iedge_original,'prewitt');
%  
% %%applying edge detection on second picture
% %so that we obtain white and black points and edges of the objects present
% %in the picture.
% 
 edge_det_pic2 = edge(Iedge_test,'prewitt');
% 
%  
% 
%initialization of different variables used
matched_data = 0;
white_points = 0;
black_points = 0;
x=0;
y=0;
l=0;
m=0;

%for loop used for detecting black and white points in the picture.
for a = 1:1:256
    for b = 1:1:256
        if(edge_det_pic1(a,b)==1)
            white_points = white_points+1;
        else
            black_points = black_points+1;
        end
    end
end

%for loop comparing the white (edge points) in the two pictures
for i = 1:1:256
    for j = 1:1:256
        if(edge_det_pic1(i,j)==1)&&(edge_det_pic2(i,j)==1)
            matched_data = matched_data+1;
        else 
        end
    end
end
%calculating percentage matching.
total_data = white_points;
total_matched_percentage = (matched_data/total_data)*100;
 
% R= corrcoef(data1,data2);
% if (R(1,1) ==1 && R(2,1)==1.0000 && R(2,1)==1.0000 && R(2,2)==1)
%     Total_r=100;
%     
% else 
%     R_2=  abs((R(1,2)-1))*100 ;
% R_3=  abs((R(2,1)-1))*100;
%   Total_r=R_2+R_3;
total_perc(ctn)=total_matched_percentage;
ctn=ctn+1;
end 
 percentage_evaluation= ((max(total_perc)-mean(total_perc)));
 fprintf('Total matched percentage is  %3.2f \n',percentage_evaluation); 
 
