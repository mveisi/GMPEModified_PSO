function [t3,disind] = dismin4(n_particle,e,p,thresh,chkcond,n_ans,ind_show, logA)

%this should calculated based on the error 
%rms objective function
thresh=std(logA);
% thresh=0.4;
%norm2(res) objective function
% thresh=40;
t1(:,1)=p(1,:);
t1(:,2)=p(2,:);
t1(:,3)=p(3,:);
t1(:,4)=p(4,:);
t1(:,5)=p(5,:);
t1(:,6)=e;
%here we sort the particle according to the error
a=sort(t1(:,6));
idum=0;
for i=1:n_particle
    for j=1:n_particle
    if (a(i)==t1(j,6))
        idum=idum+1;
        t2(idum,1)=t1(j,1);
        t2(idum,2)=t1(j,2);
        t2(idum,3)=t1(j,3);
        t2(idum,4)=t1(j,4);
        t2(idum,5)=t1(j,5);
        t2(idum,6)=t1(j,6);
        t2(idum,7)=j;
    end
    end
end
t2v=t2;
% thresh=thresh_cal(t2(:,6));
% disp(thresh)

t3(1,:)=t2(1,:);
t2(1,:)=-12345;
dis_par(1)=0;
%cal culating distance between core and all particle excluding particle
%that considered as core
ijdum=0;
for k=1:n_ans-1
    for i=1:n_particle
         if (t2(i,1) ~= -12345)
            dis_par(i) = dis_cal( t3(k,:),t2(i,:));
            dis_par_ind(i)=t2(i,7);
         else 
             dis_par(i)=100000;
         end
    end
    %calculating mean of distances and we exclude cores
    dum1=0;
    jdum=0;
    for i=1:n_particle
        if (dis_par(i) ~= 100000)
            jdum=jdum+1;
            dum1=dum1+dis_par(i);
%           dum2=dum2+dis_par(i)^2;
        end
    end
    mean_dis = dum1/jdum;
    %calculating standard deviation of distances exluding cores
    dum2=0;
    jdum=0;
    for i=1:n_particle
        if (dis_par(i) ~= 100000)
            jdum=jdum+1;
            dum2=dum2+(dis_par(i)-mean_dis)^2;
        end
    end
    std_dis=sqrt(dum2/jdum);


    %here we find the low distance particle with the
    %core k, the 7th column indicate the no of particle before sorting
    for i=2:n_particle
        if (dis_par(i) < (mean_dis - std_dis))
            ijdum=ijdum+1;
            t3r(ijdum,1)=k;
            t3r(ijdum,2)=i;
            t3r(ijdum,3)=dis_par_ind(i);
            t3r(ijdum,4)=dis_par(i);
        end
    end
    %the particle that assigned to the kth core should delete from the 
    %particles cuz a particle shouldnt connect with 2 core
    for i=1:ijdum
        for j=2:n_particle
            if (j==t3r(i,2))
                t2(j,:)=-12345;
            end
        end
    end
    %here we define a vector that contain not assigned particle number
    idum=0;
    for i=2:n_particle
        if (t2(i,1)~= -12345)
            idum=idum+1;
            dum(idum)=i;
        end
    end
    %because we sort particle on the basis of their error, the first
    %particle that didnt connect to the first cell have lowest error and
    %should considered as next core. in addition this core should have
    %small error which kept in check by using (best_particle_error + 0.4)
    %critia. if one dont use this critia there would be some core wtih big
    %error relative to the best particle. if there are no particle that
    %pass the condition, the number of core changed from input (7) to
    %number of core till this step. our algorithm work by using voroni
    %function, this function need at least three core, therefore if no
    %particle meet the condition the next best particle assigned as core to
    %increase the number of core to 3.
    if (t2(dum(1),6) < t3(1,6)+ thresh) 
        t3(k+1,:)=t2(dum(1),:);
    else
        if (k < 3)
            t3(k+1,:)=t2(dum(1),:);
        elseif (k>=3)
            n_ans=k;
            break
        end
    end
end
%in this part we connect the particle to each core 
for i=1:ijdum
    for j=1:n_particle
        if (j==t3r(i,3))
            t1(j,:)=-12345;
        end
    end
end
        
for i=1:n_particle
    if (t1(i,1) ~= -12345)
        mindis=1000;
        for j=1:n_ans
            dis=dis_cal(t3(j,:),t1(i,:))
            if (dis<=mindis)
                mindis=dis;
                mindis_ind=j;
            end
        end
    disind(i,1)=mindis;
    disind(i,2)=mindis_ind;
    end
end
for i=1:n_particle
    for j=1:ijdum
        if (i == t3r(j,3))
            disind(i,1)=t3r(j,4);
            disind(i,2)=t3r(j,1);
        end
    end
end
t3=t3(:,1:6);


