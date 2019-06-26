%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% T M FEROZ ALI
%%% 
%%% Working Correct for all DB1,2,3,4,5,9
%%%
%%% KXQDA_KPCA_Reranking2_temp2.m modified for KXQDA alone for CUHK01M1
%%% RBF kernel, KXQDA2, options.KXQDAuseEnergy = 0/1,
%%% In KXQDA: 
%%%         KinCov + (10^-7)*I1;   reg for KinCov only
%%%         eig or svd same result
%%%         If useEnergy = false; then    
%%%                         qdaDims1 = sum(latent>0.0000001);  
%%%                         qdaDims = min(qdaDims1,c-1);
%%%         alpha2 norm to be used by theory. (So use in all cases)
%%% M0_psd = projectPSD((M0+M0')/2); Throws error for complex eigen values.
%%% scores = MahDist(M0_psd, K_b'*alpha, K_a'*alpha)';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

mfilename
tic

addpath([directry '/KXQDA_KPCA_OBML/']);

%% Output parameters 
metricMethod='KXQDA';

outSetMatFilenameRoot = [metricMethod '_Results/set_mat_files/'];
outAvgMatFilenameRoot = [metricMethod '_Results/mat_files/'];
outFilenamePart = [metricMethod '_DB' num2str(sys.database)];
if sys.database == 2
        if useAllProbe == 0
            outFilenamePart = [outFilenamePart '_AP0' ];
        else
            outFilenamePart = [outFilenamePart '_AP1' ];
        end
else
    outFilenamePart = [outFilenamePart '_AP1' ];
end
outFilenamePart = [outFilenamePart '_FS' num2str(featuresetting) '_K' num2str(options.ker)];
outFilenamePart = [outFilenamePart  '_E' num2str(options.KXQDAuseEnergy) '_R' num2str(options.KXQDA_Regn)];
outFilenamePart = [outFilenamePart  '_N' num2str(normalize)];
outMatFilename = [outAvgMatFilenameRoot outFilenamePart '.mat'];

for set = 1:sys.setnum
    AllsetOutFileNames(set,:) = [outSetMatFilenameRoot outFilenamePart '_S' num2str(set,'%02.f') '.mat'];
end

if ((saveSetResults == true)&&(overWriteSetResults == false))
    %Find set to be resumed 
    resumeSet = sys.setnum+1;     %To take care when all sets already exist.
    for set = 1:sys.setnum
        if exist(AllsetOutFileNames(set,:), 'file') == 2
            continue;
        else
            resumeSet = set;
            break;
        end
    end
else
    resumeSet = 1;
end
    

%% Load data
load_features_all; % load all features.

%% Initialize cmc matrix
CMCs = zeros( sys.setnum, numperson_garalley );

for set = resumeSet:sys.setnum
    tic
    fprintf('----------------------------------------------------------------------------------------------------\n');
    fprintf('set = %d \n', set);
    fprintf('----------------------------------------------------------------------------------------------------\n');

    %% Training data
    tot = 1;
    extract_feature_cell_from_all;  % load training data
    
    if(normalize~=0)
        if(normalize==1)
            apply_normalization; % feature normalization
        elseif(normalize==2)
            apply_normalization_onGoG;
        elseif(normalize==3)
            apply_normalization2;
        else
            fprintf('Undefined normalization setting \n');
            assert(false);
        end
    end

    conc_feature_cell; % feature concatenation

    % train NFST metric learning
    camIDs = traincamIDs_set{set};
    probX = feature(camIDs == 1, :);
    galX = feature(camIDs == 2, :);
    labelsX = trainlabels_set{set};
    probXLabels = labelsX(camIDs == 1);
    galXLabels = labelsX(camIDs == 2);  
 
    %% Test data
    tot = 2;
    extract_feature_cell_from_all; % load test data
    if(normalize~=0)
        if(normalize==1)
            apply_normalization; % feature normalization
        elseif(normalize==2)
            apply_normalization_onGoG;
        elseif(normalize==3)
            apply_normalization2;
        else
            fprintf('Undefined normalization setting \n');
            assert(false);
        end
        
    end
    conc_feature_cell; % feature concatenation

    camIDs = testcamIDs_set{set};
    probY = feature(camIDs == 1, :);    
    galY = feature(camIDs == 2, :);
    labelsY = testlabels_set{set};
    labelsPr = labelsY(camIDs == 1);
    labelsGa = labelsY(camIDs == 2);
    
    if sys.database == 2
        if useAllProbe == 0
            %% Using only one image of each person
            probY = probY(1:2:end,:);
            labelsPr = labelsPr(1:2:end);  
        end            
    end

    %% KXQDA
    switch options.ker
        case 1;
            [K] = kernelMatrix('lin',[galX;probX], [galX;probX]);
            K = (K+K')/2;
            [K_a] = kernelMatrix('lin',[galX;probX], probY);
            [K_b] = kernelMatrix('lin',[galX;probX], galY);
        case 2
            [K, K_a, mu] = RBF_kernel([galX;probX], probY);
            [K_b] = RBF_kernel2([galX;probX], galY, mu);  
            K = (K+K')/2;
        case 3
            K = kernel_expchi2([galX;probX],[galX;probX]);
            K_a = kernel_expchi2([galX;probX], probY);
            K_b = kernel_expchi2([galX;probX], galY);
        case 4
            [K] = kernelMatrix('chi2a',[galX;probX], [galX;probX]);
            K = (K+K')/2;
            [K_a] = kernelMatrix('chi2a',[galX;probX], probY);
            [K_b] = kernelMatrix('chi2a',[galX;probX], galY);
        case 5
            [K] = kernelMatrix('chi2b',[galX;probX], [galX;probX]);
            K = (K+K')/2;
            [K_a] = kernelMatrix('chi2b',[galX;probX], probY);
            [K_b] = kernelMatrix('chi2b',[galX;probX], galY);
        otherwise
            error(['Unsupported kernel ' options.ker])
    end
    
    [alpha, M0] = KXQDA2(galX, probX, galXLabels, probXLabels, K, options);
    %[alpha, M0] = KXQDA3(galX, probX, galXLabels, probXLabels, K, options);
    
    M0_psd = projectPSD((M0+M0')/2); 
   
    if sys.database ~= 3
        % single shot matching
        scores = MahDist(M0_psd, K_b'*alpha, K_a'*alpha)';
        
        CMC = zeros( numel(labelsGa), 1);
        for p=1:numel(labelsPr)
            score = scores(p, :);
            [sortscore, ind] = sort(score, 'ascend');
            
            correctind = find( labelsGa(ind) == labelsPr(p));
            CMC(correctind:end) = CMC(correctind:end) + 1;
        end
        CMC = 100.*CMC/numel(labelsPr);
        CMCs(set, :) = CMC;
        
    else
        % multi shot matching
        
        labelsPr1 = labelsPr(1:2:size(probY, 1), 1);
        labelsPr2 = labelsPr(2:2:size(probY, 1), 1);
        labelsGa1 = labelsGa(1:2:size(galY, 1), 1);
        labelsGa2 = labelsGa(2:2:size(galY, 1), 1);
        
        scores1 = MahDist(M0_psd, K_b(:,1:2:end)'*alpha, K_a(:,1:2:end)'*alpha)';
        scores2 = MahDist(M0_psd, K_b(:,2:2:end)'*alpha, K_a(:,1:2:end)'*alpha)';
        scores3 = MahDist(M0_psd, K_b(:,1:2:end)'*alpha, K_a(:,2:2:end)'*alpha)';
        scores4 = MahDist(M0_psd, K_b(:,2:2:end)'*alpha, K_a(:,2:2:end)'*alpha)';
  
        scores = scores1 + scores2 + scores3 + scores4;
        
        CMC = zeros( numel(labelsGa1), 1);
        for p=1:numel(labelsPr1)
            score = scores(p, :);
            [sortscore, ind] = sort(score, 'ascend');
            
            correctind = find( labelsGa1(ind) == labelsPr1(p));
            CMC(correctind:end) = CMC(correctind:end) + 1;
        end
        CMC = 100.*CMC/numel(labelsPr1);
        CMCs(set, :) = CMC;
    end
    toc
    
    fprintf(' Rank1,  Rank5, Rank10, Rank15, Rank20\n');
    fprintf('%5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%\n\n', CMC([1,5,10,15,20]));
    clear camIDs probX galX probX galLabels probLabels XQDAresult
    
    %Saving each set results
    if((write2file == true)&&(saveSetResults == true))
        CMC = single(CMC);
        eval(['save ' AllsetOutFileNames(set,:) ' CMC']);
    end     

end  

if((write2file == true)&&(saveSetResults == true))  %Saved set results exist only if saveSetResults = true
    %Loading and Accumulating results of all sets
    for set =1:sys.setnum
        eval(['load ' AllsetOutFileNames(set,:)]);
        CMCs(set, :) = CMC;
    end
end
  
toc

clear interSampleDistTst  num_test finalScore sortScore sortIndex cmcCurrent

fprintf('----------------------------------------------------------------------------------------------------\n');
fprintf('  File: %s\n', mfilename);
fprintf('  Mean Result DB%d Norm%d featuresetting %d \n', [sys.database normalize featuresetting]);
fprintf('----------------------------------------------------------------------------------------------------\n');
clear set;

CMCmean = single(mean( squeeze(CMCs(1:sys.setnum , :)), 1));
fprintf(' Rank1,  Rank5, Rank10, Rank15, Rank20\n');
fprintf('%5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%, %5.2f%% \n', CMCmean([1,5,10,15,20]) );

clear num_gallery;

if(write2file == true)
    CMCmeanR1_20 = CMCmean([1,5,10,15,20]);
    eval(['save ' outMatFilename ' CMCmean CMCmeanR1_20 -v7.3']);
     
    %Save to csv
    col1 = [sys.database useAllProbe featuresetting options.ker options.KXQDAuseEnergy options.KXQDA_Regn CMCmeanR1_20];
    c3 = clock; dateTime = [num2str(c3(3)) ':' num2str(c3(4)) ':' num2str(c3(5)) ':' num2str(c3(6)) ':'];
    %dlmwrite(outCsvFilename,col1,'precision',4,'-append');
    fprintf(fileID,'%d %d %d %d %d %d %.2f %.2f %.2f %.2f %.2f %s \n',col1,dateTime);
end
