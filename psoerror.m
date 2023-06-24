
function  e = psoerror(logA,R,logR,M,pos,nA,R01)
%this function calculate objective function for pso
ind_rms=1;
e=0;
for i=1:nA
    if (R01 <= R(i))
        Rs2=(R(i) /R01);
        synth = pos(1)*M(i) + pos(2)*log10(R01) + pos(3)*log10(Rs2)*1 +pos(4)*R(i) + (pos(5));
    else
        synth = pos(1)*M(i) + pos(2)*log10(R(i))+pos(4)*R(i) + (pos(5));
    end  
    val= synth -logA(i);
    e=e+val*val;
end
if (ind_rms == 1)
    e = sqrt(e/nA);
else
    e=sqrt(e);
end
end