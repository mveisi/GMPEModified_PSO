function [pos_ls, error_of_pos_ls] = wls(Ulow,Uup,R,M,logR,R01,logA)

if (R01 >= max(R))
    weight=0;
    ind_kink = 1;
else
    weight=1;
    ind_kink = 2;
end
initial_value_ind = 1;
initial_value= [1.4; -1.0; -1.0; -0.012; -6.7]; %starting value for linear
                                                %imposed by a weight
if (ind_kink == 2) 
    for i=1:length(M)
        if (R01 <= R(i))
            Rs2=(R(i)/R01);
            jacob(i,1)=M(i);
            jacob(i,2)=log10(R01);
            jacob(i,3)=log10(Rs2);
            jacob(i,4)=R(i);
            jacob(i,5)=1.0;
        else
            jacob(i,1)=M(i);
            jacob(i,2)=log10(R(i));
            jacob(i,3)=0.000;
            jacob(i,4)=R(i);
            jacob(i,5)=1.0;
        end
    end
else
    for i=1:length(M)
            jacob(i,1)=M(i);
            jacob(i,2)=log10(R(i));
            jacob(i,3)=R(i);
            jacob(i,4)=1.0;
    end
end
w=eye(5)*weight;

G=jacob;
b=logA;
if (weight > 0)
    idum=0;
    for i=1+length(M):length(M)+5
        idum=idum+1;
        G(i,:)=w(idum,:);
    end
    
    idum = 0;
    for i=1+length(M):length(M)+5
        idum = idum +1;
        if (initial_value_ind == 1)
            b(i)=initial_value(idum);
        else
            b(i) = 0;
        end
    end
end
if (ind_kink == 2) 
    pos_ls=(G'*G)\G'*b;
else
    pos_ls=(G'*G)\G'*b;
    pos_ls_cp = pos_ls;
    pos_ls(1:2) = pos_ls_cp(1:2);
    pos_ls(3) = 0.0;
    pos_ls(4) = pos_ls_cp(3);
    pos_ls(5) = pos_ls_cp(4);
end



e=0;
for i=1:length(M)
        
   if (R01 <= R(i))
        Rs2=(R(i) /R01);
        synth = pos_ls(1)*M(i) + pos_ls(2)*log10(R01) + pos_ls(3)*log10(Rs2)*1 +pos_ls(4)*R(i) + (pos_ls(5));
   else
        synth = pos_ls(1)*M(i) + pos_ls(2)*log10(R(i))+pos_ls(4)*R(i) + (pos_ls(5));
   end
              
   val= synth -logA(i);
   e=e+val*val;
end

pos_ls(6) = sqrt(e/length(M));
pos_ls(7)=sqrt(e);

error = inv(G'*G);
error_of_pos_ls= diag(error);
end

