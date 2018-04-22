import numpy as np
from math import *
import TOM_SAPS
from removeBias2 import removeBias

def calculateQvSingleAsset(ALG_OPTION, BIAS_METHOD, rfObjects, irObjects, sigRF, sigIR, gatingK, nRF, nIR, indexRF, indexIR):
	
	indexRF = indexRF[np.logical_not(np.isnan(indexRF))]
	indexIR = indexIR[np.logical_not(np.isnan(indexIR))]
	print("indexRF = ", indexRF)
	print("indexIR = ", indexIR)
	rfObjects = rfObjects[:,indexRF]
	irObjects = irObjects[:,indexIR]
	print("rfObjectsw = ", rfObjects)
	print("irObjects = ", irObjects)
	
	lethalObjNum = 0
	score = 0
	
	if ALG_OPTION == 1:
		# Bias Removal
		bias = removeBias(rfObjects, irObjects, BIAS_METHOD)
		x_unbiased = rfObjects + bias  ## TEST broadcast
		
		# Correlation
		## TBD
		M = MahalDist(x_ub, x_ir, SigRF, SigIR);
		P = ModMunkres2(M')
		indrff = indrf
        
		Nrff = Nrf;

	elif ALG_OPTION == 2:
		pass
	elif ALG_OPTION == 3:
		pass
		
	return (lethalObjNum, score)
	
"""

indrf = indrf(~isnan(indrf));
rf_objects = rf_objects(:,indrf);

indir = indir(~isnan(indir));
ir_objects = ir_objects(:,indir);

switch(ALG_OPTION)
    case 1
        %Bias removal:
        bias = remove_bias(rf_objects, ir_objects, BIAS_METHOD);
        x_ub(1,:) = rf_objects(1,:) + bias(1);
        x_ub(2,:) = rf_objects(2,:) + bias(2);
        
        %Now do correlation:
        M = MahalDist(x_ub, x_ir, SigRF, SigIR);
        P = ModMunkres2(M');
        indrff = indrf;
        Nrff = Nrf;
    case 2
        %Bias removal:
        bias = remove_bias(rf_objects, ir_objects, BIAS_METHOD);
        x_ub(1,:) = rf_objects(1,:) + bias(1);
        x_ub(2,:) = rf_objects(2,:) + bias(2);
        [M,D] = MahalDist3(x_ub, ir_objects, SigRF, SigIR, GatingK);
        indrff = indrf;
        P = zeros(Nir,1);
        PP = ModMunkres2(M');
        %Now break any associations that are too far:
        j = 0;
        for i = 1 : Nir
            j = j + 1;
            if PP(j) > 0 && PP(j) < BIG_L
                P(i) = PP(j);
            end
        end
    case 3
        if BIAS_METHOD == 4
            % Actually do things the right way, with multiple
            % bias combinations to find the right one:
            bias = remove_bias2(rf_objects, ir_objects, SigRF, SigIR);
        else
            %Bias removal:
            bias = remove_bias(rf_objects, ir_objects, BIAS_METHOD);
            
        end
        %% TODO: double check - had to reverse the signs in the bias removal calculation
        x_ub(1,:) = rf_objects(1,:) - bias(1);
        x_ub(2,:) = rf_objects(2,:) - bias(2);
        
        %%%%%%%%%%%%%%%%%%%%%%
%         for i = 1:Nrf
%             index = indrf(i);
%             plot(x_ub(1,index), x_ub(2,index), 'go', 'MarkerSize', 30);
%             text(x_ub(1,index), x_ub(2,index), num2str(index));
%             [Trf,d] = eig(SigRF);
%             if (d(1,1) > d(2,2))
%                 sig_rf_maj = sqrt(d(1,1));
%                 sig_rf_min = sqrt(d(2,2));
%             else
%                 sig_rf_maj = sqrt(d(2,2));
%                 sig_rf_min = sqrt(d(1,1));
%             end
%             
%             [x,y] = draw_ellipse(x_ub(1,index), x_ub(2,index), sig_rf_maj, sig_rf_min, Trf);
%             plot(x,y,'g:');
%         end
        %%%%%%%%%%%%%%%%%%%%%%%
        
        %Gating
        [M,D] = bhattacharyya(x_ub, ir_objects, SigRF, SigIR, GatingK);
%         [M2,D2] = mahal_dist3(x_ub, ir_objects, SigRF, SigIR, GatingK);  %% TODO  check this
        %                     [M2,D2] = mahal_dist3(x_ub, x_ir, SigRF, SigIR, GatingK);
        %                     [M,D] = bhattacharyya(x_ub, x_ir, SigRF, SigIR, GatingK);
        out = zeros(Nrf,1);
        PM = [];
        for i = 1 : Nrf
            if min(M(i,:)) < BIG_L
                PM = [PM; M(i,:)];
                out(i) = 1;
            end
        end
        indrff = indrf(out == 1);
        Nrff = length(indrff);
        P = zeros(Nir,1);
%         if ~isempty(PM)
%             PPM = [];
%             indirr = zeros(Nir,1);
%             for i = 1 : Nir
%                 if min(PM(:,i)) < BIG_L
%                     PPM = [PPM PM(:,i)];
%                     indirr(i) = 1;
%                 end
%             end
%             if ~isempty(PPM)
%                 PP = mod_munkres2(PPM');
%                 %Now break any associations that are too far:
%                 j = 1;
%                 for i = 1 : Nir
%                     if indirr(i) && PP(j) < BIG_L
%                         if PP(j) > 0
%                             P(i) = PP(j);
%                         end
%                         j = j + 1;
%                     end
%                 end
%             end
%         end
        out = zeros(Nir,1);

        if ~isempty(PM)
            PPM = [];
            indirr = zeros(Nir,1);
            for i = 1 : Nir
                if min(PM(:,i)) < BIG_L
                    PPM = [PPM PM(:,i)];
                    out(i) = 1;
                end
            end
            indirr = indir(out == 1);
            if ~isempty(PPM)
                PP = mod_munkres2(PPM');
                %Now break any associations that are too far:
                j = 1;
                for i = 1 : Nir
                    if out(i) && PP(j) > 0
                        if (PM(PP(j),i) < BIG_L)
                            if PP(j) > 0
                                P(i) = PP(j);
                            end
                        end    
                        j = j + 1;
                        
                    end
                end
            end
        end
end

%Final bookkeeping:
lethal_obj_num = 0;

score = 0;
for i = 1 : Nir
    if P(i) > 0
        Assign(i) = indrff(P(i));
        if (Assign(i) == 1)  %% if the IR object was assigned to object 1, then it is assumed to be the object of interest
            lethal_obj_num = indrff(P(i));
        end
        score = score + PM(P(i), i);
    else
        Assign(i) = 0;
    end
end
	"""
	
if __name__ == "__main__":
	ALG_OPTION = 3
	BIAS_METHOD = 6
	rfObjects = np.array([ [0, 1, 2, 3, 0], [1, -1, 0, 3, 2]])
	#x2 = np.array([ [ 0, 10, 11, 2], [11, -11, 4, 0]])
	#test_bias = np.array( [ [1,],[1,] ])
	irObjects = np.array([ [0, np.nan, np.nan, np.nan, np.nan, 1, 2, 3, 4], [1, np.nan, np.nan, np.nan, np.nan, -1, 0, 3, 5]])
	sigRF = np.array( [[4, 0],[0,4]])
	sigIR = np.array( [[1,0],[0,1]])
	gatingK = 5
 	nRF = 5
 	nIR = 5
 	indexRF = np.array([0,1,2,3,4])
 	indexIR = np.array([0,np.nan,5,6,7,8])
 	(lethalObjNum, score) = calculateQvSingleAsset(ALG_OPTION, BIAS_METHOD, rfObjects, irObjects, sigRF, sigIR, gatingK, nRF, nIR, indexRF, indexIR)