
function  [yobs,ysyn] = psoerror_for_obs_synth(logA,R,logR,M,pos,nA,R01)
%this function calculate objective function for pso
% R01=30;
R02=40;
for i=1:nA
        
              if (R01 <= R(i))
                  Rs2=(R(i) /R01);
                  synth = pos(1)*M(i) + pos(2)*log10(R01) + pos(3)*log10(Rs2)*1 +pos(4)*R(i) + (pos(5));
              else
                  synth = pos(1)*M(i) + pos(2)*log10(R(i))+pos(4)*R(i) + (pos(5));
              end
% [ synth ] = synth_cal(pos,M(i),R(i),R01,R02,1 );
              
    yobs(i)=logA(i)-pos(1)*M(i);
    syn(i)=synth;
    ysyn(i)=syn(i)-pos(1)*M(i);
end

end