
function  [valout,e] = errorplot(logA,R,logR,M,pos,nA,R01)

e=0;
% R01=30;
R02=40;
for i=1:nA
        
              if (R01 <= R(i))
                  Rs2=(R(i) /R01);
                  synth = pos(1)*M(i) + pos(2)*log10(R01) + pos(3)*log10(Rs2)*1 +pos(4)*R(i) + (pos(5));
              else
                  synth = pos(1)*M(i) + pos(2)*log10(R(i))+pos(4)*R(i) + (pos(5));
              end
%     [ synth ] = synth_cal(pos,M(i),R(i),R01,R02,1 );
      
    val= synth -logA(i);
    valout(i,1)=synth -logA(i);
    e = e + val*val;
    valout(i,2)=R(i);
    valout(i,3)=M(i);
    
end

difr=max(R)/nA;
r=min(R);
for m=1:nA
    r=r+difr;
    
              if (R01 <= r)
                  Rs2=(r /R01);
                  synth2 = pos(1)*M(i) + pos(2)*log10(R01) + pos(3)*log10(Rs2)*1 +pos(4)*r + (pos(5));
              else
                  synth2 = pos(1)*M(i) + pos(2)*log10(r)+pos(4)*r + (pos(5));
              end
% [ synth2 ] = synth_cal(pos,M(i),r,R01,R02,1 );
    valout(m,4)=synth2;
end


%valout=[valout' R];
% save('er-data.txt','valout','-ascii')

e = sqrt(e/nA);
%vectorize
%val = pos(1)*M + pos(2)*R + pos(3)*logR + pos(4) - logA;
% e =sum(val.*val);
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here
end