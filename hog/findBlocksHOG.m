%***************************
%*  HOG Feature Extractor
%*  Mahmoud Farshbaf Doustar
%* 
%*  2010,3,10
%*  References :
%*  - Dalal, N., Triggs, B.: Histograms of Oriented Gradients for Human detection. In: CVPR 2005 (2005)
%*  - D. Lowe. Distinctive Image Features from Scale-Invariant Keypoints. IJCV, 60(2):91?110, 2004.
%***************************
%
%   window : is the input (a sub window (ROI) of the image to process)
%   signed : choosing between signed or unsigned gradient (0 unsigned , 1 signed)
%   cellSize: the size of the cell(in pixel) involved in histogram calculation
%   binSize : the size of the orientation bin ( if gradient is signed -0 to 360, the bin size of 10 means : 0-35,36-71,....,328-359
%   option: define the output format that could be 'vector' , 'matrix'
%   showWindow : determine  'showWindow' for debug or 'noWindow'
%
%******************************

function [blocks]=findBlocksHOG (window,signed,cellSize,binSize,option,showWindow)


I=window;


[Fx,Fy]=myGradient(im2double(I));

signed_ang=atan2(Fy,Fx)*180/pi;


if(signed)
    ang=signed_ang;
    total_Ang=360;
else
    unsigned_ang=signed_ang;
    unsigned_ang(signed_ang(:)<0)=signed_ang(signed_ang(:)<0)+180;
    ang=unsigned_ang;    
    total_Ang=180;
end

val=sqrt(Fy.^2+Fx.^2);

% do gaussian spatial weighting to magnitude of each point
h = fspecial('gaussian', [cellSize cellSize], cellSize);
val=imfilter(val,h);


[ROIh,ROIw ]=size(signed_ang);

cellNumI=floor(ROIh/cellSize);
cellNumJ=floor(ROIw/cellSize);

OrientationBin=zeros(cellNumI,cellNumJ,total_Ang/binSize);


for cellI=1:cellNumI
    for cellJ=1:cellNumJ
        
        for bin=1:total_Ang/binSize
            startI=(cellI-1)*cellSize+1;
            endI=(cellI)*cellSize;
            startJ=(cellJ-1)*cellSize+1;
            endJ=(cellJ)*cellSize;
            
            A=zeros(endI-startI+1,endJ-startJ+1);
            
            for i=startI:endI
                for j=startJ:endJ
                    
                    if((ang(i,j)>=(bin-1)*binSize+1)&&(ang(i,j)<(bin)*binSize))
                    A(i-startI+1,j-startJ+1)=1;
                    end
                    if(bin>1)
                        if((ang(i,j)>=(bin-2)*binSize+1+binSize/2)&&(ang(i,j)<(bin-1)*binSize))
                            A(i-startI+1,j-startJ+1)=1-abs(ang(i,j)-(bin*binSize-binSize/2))/binSize;
                        end
                    end
                    if(bin<total_Ang/binSize)
                        if((ang(i,j)>=(bin)*binSize+1)&&(ang(i,j)<(bin+1)*binSize-binSize/2))
                            A(i-startI+1,j-startJ+1)=1-abs(ang(i,j)-(bin*binSize-binSize/2))/binSize;
                        end
                    end
                    
                    
                    
                end
            end
                
            OrientationBin(cellI,cellJ ,bin) = sum(sum( A .* val(startI:endI,startJ:endJ)));
            
        
        end
        
                
        if(strcmp('showWindow',showWindow))
            subplot(cellNumI,cellNumJ, (cellI-1)*cellNumJ+cellJ);
            vector=permute(OrientationBin(cellI,cellJ,:),[3,2,1]);
            bar(vector);
            %remove xticks and yticks
            set(gca,'xtick',[],'ytick',[]);
        end
        
        
    end
end

% normalizing feature vectors by grouping in a block

%overlapping blocks: block=2 neighbor  cells
cellInBlock=4;
blockNumI=cellNumI-1;
blockNumJ=cellNumJ-1;


OrientationBinBlocks=zeros(blockNumI,blockNumJ,cellInBlock*total_Ang/binSize);
for blockI=1:blockNumI
    for blockJ=1:blockNumJ
        %create Block
        blockVector=zeros(1,cellInBlock*total_Ang/binSize);
        for i=1:2
            for j=1:2
                cellI=(blockI-1)+i;
                cellJ=(blockJ-1)+j;
                vector=permute(OrientationBin(cellI,cellJ,:),[3,2,1]);
                %1    2
                %3    4
                cellNumInBlock=((i-1)*2+j);
                
                blockVector((cellNumInBlock-1)*(total_Ang/binSize)+1:(cellNumInBlock)*(total_Ang/binSize))=vector;
                
            end
        end
        %block creating end
        
        %normalizing blocks
        %%%L1-norm
            %normBlockVector=blockVector ./ sum(blockVector);
         %%%%%%%
         
        %%%L2-Hys
        normBlockVector=blockVector ./ sum(blockVector);
        normBlockVector(normBlockVector(:)>0.2)=0.2;
        normBlockVector=normBlockVector ./ sum(normBlockVector);
        
        
        OrientationBinBlocks(blockI,blockJ,:)=normBlockVector;
        
    end
end

if(strcmp('matrix',option))
blocks=OrientationBinBlocks;
else
    blocks=zeros(1,size(OrientationBinBlocks,1)*size(OrientationBinBlocks,2)*size(OrientationBinBlocks,3));
    for i=1:size(OrientationBinBlocks,1)
        for j=1:size(OrientationBinBlocks,2)
            for k=1:size(OrientationBinBlocks,3)
            
            blocks((i-1)*size(OrientationBinBlocks,2)*size(OrientationBinBlocks,3)+(j-1)*size(OrientationBinBlocks,3)+k)=OrientationBinBlocks(i,j,k);
            end
        end
        
    end
end