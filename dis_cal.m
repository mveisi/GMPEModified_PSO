function [ dis_par ] = dis_cal( a,b )

dum1=(a(1,1)-b(1,1))^2;
dum2=(a(1,2)-b(1,2))^2;
dum3=(a(1,3)-b(1,3))^2;
dum4=(a(1,4)-b(1,4))^2;
dum5=(a(1,5)-b(1,5))^2;
dis_par=sqrt(dum1+dum2+dum3+dum4+dum5)
end

