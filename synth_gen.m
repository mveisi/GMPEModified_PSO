function [logA_syn,pos_syn] = synth_gen(Ulow,Uup,R,M,logR,R01,logA, er_logA)

%here we going to make a synthetic logA to test our method
%er_logA is the imposed error to the synthetic data
max_er=-1; %for imposing max error in all dis and magnitudes, if <0
           %the error is randomly distribute
% er_logA=20; % percent of maximum error
ind_kink=2; %1 for attenuation relationship without kink, 2 for kink
            %tip: dont change this just put the value of 
            %kink in the main code greater than the maximum distance 
            %existed in your dataset


%%%initial value for synthetic
pos(1) = 1.4;
pos(2) = -1.0;
pos(3) = -1.0;
pos(4) = -0.0150;
pos(5) = -6.5;

%%%%% if you want to have different attenuation parameters for distance 
%%%%greater than the kink value
pos2(1) = 1.4;
pos2(2) = -1.0;
pos2(3) = -1.0;
pos2(4) = -0.0150;
pos2(5) = -6.5;


pos_syn=pos;
pos_syn2=pos2;
if (ind_kink == 2)
    for i=1:length(M)
    rand_ind = rand;
            if (rand_ind <= 0.5)
                if (R01 <= R(i))
                    Rs2=(R(i)/R01);
                    synth(i) = pos(1)*M(i) + pos(2)*log10(R01) + pos(3)*log10(Rs2)*1 +pos(4)*R(i) + (pos(5));
                else
                    synth(i) = pos(1)*M(i) + pos(2)*log10(R(i))+pos(4)*R(i) + (pos(5));
                end
            else
                if (R01 <= R(i))
                    Rs2=(R(i)/R01);
                    synth(i) = pos2(1)*M(i) + pos2(2)*log10(R01) + pos2(3)*log10(Rs2)*1 +pos2(4)*R(i) + (pos2(5));
                else
                    synth(i) = pos2(1)*M(i) + pos2(2)*log10(R(i))+pos2(4)*R(i) + (pos2(5));
                end 
            end 
    end
end
if (ind_kink == 1)
    for i=length(M)
        rand_ind = rand;
        if (rand_ind <= 0.5)
            synth(i) = pos(1)*M(i) + pos(2)*log10(R(i))+pos(4)*R(i) + (pos(5));
        else
            synth(i) = pos2(1)*M(i) + pos2(2)*log10(R(i))+pos2(4)*R(i) + (pos2(5));
        end
    end
end

if (er_logA > 0)
    for i=1:length(M)
        rand_number = rand();
        if (rand_number < 0.05)
            rand_number = -rand_number * 2;
        else
            rand_number = (rand_number - 0.5) * 2;
        end
        rand_number
        synth2(i)=synth(i)+(rand_number*er_logA*synth(i)/100);
    end
end
if (max_er > 0)
    for i=1:length(M)
        synth2(i)=synth(i)+(rand*max_er);
    end
end

scatter(1:length(synth),synth-synth2)

logA_syn=synth2';
end

