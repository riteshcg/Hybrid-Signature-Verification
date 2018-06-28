%% 
clc;
clear all;
close all;
warning off all;

%% load test data 
[file,path]=uigetfile('*.hwr*','Load file');

filename=strcat(path,file);
count=1;
[ext,name,path]=fileparts(file);
for k = 10:length(name)
 req(count)=name(k);
 count=count+1;
end
% filename = 'F:\Payement receipt\13- Signature\FINAL CODE\data_online\35\NFI-00304035.hwr';
startRow = 2;
formatSpec = '%1s%5f%6f%6f%6f%f%[^\n\r]';
fileID = fopen(filename,'r');
 dataArray = textscan(fileID, formatSpec, 'Delimiter', '', 'WhiteSpace', '', 'HeaderLines' ,startRow-1, 'ReturnOnError', false);
 dataArray{1} = strtrim(dataArray{1});
 fclose(fileID);
value2 = dataArray{:, 4}';
value3 = dataArray{:, 5}';
value4 = dataArray{:, 6}';
clearvars filename startRow formatSpec fileID dataArray ans;
test_case=[value2(:,1:100) value3(:,1:100) value4(1:100) ];
 subplot 131,plot(value2,'r')
 subplot 132,plot(value3,'r')
 subplot 133,plot(value4,'r')
 
%% GENUINE
inside_count=1;
rem_pat=dir(strcat('ONLINE GENUINE','\',req));
for j=3:length(rem_pat) 
      
%     path=(rem_pat(i).name);
    image =rem_pat(j).name ;
    full_path=strcat(['ONLINE GENUINE','\',req,'\',image]);
    startRow = 2;
formatSpec = '%1s%5f%6f%6f%6f%f%[^\n\r]';
fileID = fopen(full_path,'r');
dataArray = textscan(fileID, formatSpec, 'Delimiter', '', 'WhiteSpace', '', 'HeaderLines' ,startRow-1, 'ReturnOnError', false);
dataArray{1} = strtrim(dataArray{1});
 fclose(fileID);
 
value2  = dataArray{:, 4}';
value3  = dataArray{:, 5}';
value4  = dataArray{:, 6}';
features_genuine(inside_count,:)=[ value2(1:100) value3(1:100)   value4(1:100) ];
clearvars filename startRow formatSpec fileID dataArray ans;
 
 inside_count=inside_count+1;
%   subplot 221,plot(value1,'r')
%  subplot 222,plot(value2,'r')
%  subplot 223,plot(value3,'r')
%  subplot 224,plot(value4,'r')
%  pause(0.5)
%  close all
 
 end 
%%   FAKE 
  
 inside_count=1;
rem_pat=dir(strcat('ONLINE FAKE','\',req));
for j=3:length(rem_pat) 
      
%     path=(rem_pat(i).name);
    image =rem_pat(j).name ;
    full_path=strcat(['ONLINE FAKE','\',req,'\',image]);
    startRow = 2;
formatSpec = '%1s%5f%6f%6f%6f%f%[^\n\r]';
fileID = fopen(full_path,'r');
dataArray = textscan(fileID, formatSpec, 'Delimiter', '', 'WhiteSpace', '', 'HeaderLines' ,startRow-1, 'ReturnOnError', false);
dataArray{1} = strtrim(dataArray{1});
 fclose(fileID);
 
value2  = dataArray{:, 4}';
value3  = dataArray{:, 5}';
value4  = dataArray{:, 6}';
features_fake(inside_count,:)=[ value2(1:100) value3(1:100)   value4(1:100) ];
clearvars filename startRow formatSpec fileID dataArray ans;
 
 inside_count=inside_count+1;
%   subplot 221,plot(value1,'r')
%  subplot 222,plot(value2,'r')
%  subplot 223,plot(value3,'r')
%  subplot 224,plot(value4,'r')
%  pause(0.5)
%  close all
 
 end 
 %% label 
 genuine=features_genuine';
 fake=features_fake';
 features=[genuine  fake];
 %  
  [m,n]=size(genuine);
  [p,q]=size(fake);
  label_gen=ones(1,n);
  label_com=zeros(1,q);
  final_gen=[label_gen label_com];
  final_fake=[1- final_gen];
  %
  labels_total=[final_gen ; final_fake];
  
  %
  net = patternnet(50);
 [net,tr] = train(net, features, labels_total);
%    
  nntraintool 
  plotperform(tr)
  testX1 = features(:,tr.testInd);
  testT1 = labels_total(:,tr.testInd);
  testY1 = net(testX1);
  testIndices = vec2ind(testY1);
   pause(0.002)
  tp=test_case';
%% Checking 
for i=1:size(genuine,2)
   test_cat= genuine(:,i);
   R=corrcoef(test_cat,tp);
   out(1,i)=R(1,1);
   out(2,i)=R(1,2);
   out(3,i)=R(2,1);
   out(4,i)=R(2,2);
    
end 
 out(out~=1)=0;
 total_out=mean(mean(out));
 if total_out>0.5
     disp('GENUINE')
 else 
     disp('FAKE')
 end 