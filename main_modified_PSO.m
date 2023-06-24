%% report bug or question to m.veisi9687@gmail.com





clear all
format long

fid = fopen('inputarguments');

% Read the first line
line = fgetl(fid);

% Loop until the end of the file
idum = 0;
while ischar(line)
    % Do something with the line
    idum = idum +1;
    if (idum == 1)
        inp_file = line;
    elseif (idum == 2)
        synthetic_run = line;
        if (synthetic_run == "True")
            synth_ind = 1;
        else
            synth_ind = 0;
        end
    elseif (idum ==3)
        synthetic_error = line;
        er_logA = str2double(synthetic_error);
    elseif (idum == 4)
        real_data_run = line;
        if (real_data_run == "True")
            synth_ind = 0;
        else
            synth_ind = 1;
        end
    elseif (idum == 5)
        kink_value = line;
        R01 = str2double(kink_value);
    elseif (idum == 6)
        cut_value = line;
        cut_r = str2double(cut_value);
    elseif (idum == 7)
        N_subspace = line;
        n_ans = str2double(N_subspace);
    else
        output_folder = line;
    end 
    
    % Read the next line
    line = fgetl(fid);
end

% Close the file
fclose(fid);

%%%%%%%%%%%%%%
rng(100)  %fixing random seed
%====s
n_iter_all = 1; %index for creating the output folder and possible looping 
                %of whole script.

% cut_r = 70;
% R01=40; %% kink value
% synth_ind=0;    %equal to 1 for synthetic run
chkstd_ind=1;   %equal to 1 for rearranging clusters
n_particle=1000; %number of particles
w=0.03;           %inertia coefficient
c1=0.2;            %learning coefficient itself
c2=0.1;            %learning coefficient for gbest in FSS
c3=0.1;            %learning coefficient for gbest in SSS (smaller because 
                   %we dont want to particles in each clusters have a fast 
                   %movement into the cluster core gbest as the cluster 
                   %could change in SSS 
p = zeros(5,n_particle);   %positions
q = zeros(5,n_particle);   %velocities
r = zeros(5,n_particle);   %ibest
n_update=5;     %number of iteration which cluster update occur
chkcond=0.45;   %imposing a manual threshhold, this do nothing in this code
% n_ans=3;        %number of solutions which code going to output
n_iter=5;      %number of FSS iteration control the amount of aggregation
                %before entering the SSS. increasing this may result in 
                %similar solutions, lowering it may result in unreliable
                %solution, the purpose of FSS is to make the particle
                %position more relevent and remove the particles random
                %properties which generated as the PSO initilize.
if (synth_ind == 1)
    n_iter = 10;
else
    n_iter = 5;% we have a very low std real dataset so i reduce the 
                % first step search iteration number to avoid highly
                % aggregated particles enters the SSS. you can change this
                % according to your dataset quality. 
end
    
n_iter2=74;     % number of iteration in SSS, should satisfy mod(n_iter2, n_update) = n_update -1


% %data northwest===========================
load(inp_file) %%before_adding new data from station id
R=data_NWIran(:,1);     %distance in km
M=data_NWIran(:,2);     %magnitudes
logA=(data_NWIran(:,3));    %log(amp)
%==========================================
%cuting the data for a specific distance
ind=find(R<cut_r);     
R=R(ind);M=M(ind);logA=logA(ind);
[nA dum]=size(logA);
logR=log10(R);
%Initialize the particle's position with a uniformly distributed random vector: xi ~ U(blo, bup)
%Initialize the particle's best known position to its initial position: pi : xi

%initializing the pso particle boundary  
%in pso algorithm we can assign our solution to be bounded by two number
%these number can be found from previous work.
%here we want to calculate 5 parameter that fit the data to a line with 
% this equation: logA = pos(1)*M(i) + pos(2)*log10(Rs) + pos(3)*log10(Rs2)*heaviside(R(i)-R0) +pos(4)*R(i) + (pos(5));
%where pos vector in this eq is our solution.
%===========================================================
%here we choose the lower and upper bound of unknown parameters parameter
% indexing: 1 == a, 2 == b1, 3 == b2, 4 == c, 5 == d.
Ulow(1)= 0.5;
Uup(1)= 2;
Ulow(2)= -2.5;
Uup(2)= -0.5;
Ulow(3)= -2.5;
Uup(3)= 0.5;
Ulow(4)= -0.03;
Uup(4)= -7.37e-04;
Ulow(5)= -8.5;
Uup(5)= -4.5;
%===========================================================

%===================synthetic run block========================================
% er_logA = 20;
if (synth_ind == 1) 
    [logA_syn,pos_syn] = synth_gen(Ulow,Uup,R,M,logR,R01,logA, er_logA);
    [pos_ls, error_of_pos_ls] = wls(Ulow,Uup,R,M,logR,R01,logA_syn);
    logA=logA_syn;
    data_syn(:, 1) = R;
    data_syn(:, 2) = M;
    data_syn(:, 3) = logA_syn;
    fl_name = strcat(output_folder, '/', 'data_syn.dat');
    save(fl_name, 'data_syn', '-ascii');
    
else
    [pos_ls, error_of_pos_ls] = wls(Ulow,Uup,R,M,logR,R01,logA, synth_ind);
end
%=========================================
%initializing pso parameter
g = zeros (1,5);   %swarm best known poition
for i=1:n_particle
    for j=1:5
        p(j,i)= Ulow(j) + rand*(Uup(j) - Ulow(j));   %random position
    end
    r(1:5,i)= p(1:5,i);   %first best known position for each particle. which is equal to position in iter 1
    % if f(pi) < f(g) then
    %update the swarm's best known  position: g : pi
    if (i == 1)
        g= p(1:5,i);  %first random gbest
    end
    %here we find the gbest from all particles.
    if (psoerror(logA,R,logR,M,p(1:5,i),nA,R01)< psoerror(logA,R,logR,M,g,nA,R01))
        g= p(1:5,i);  
    end
    %Initialize the particle's velocity: vi ~ U(-|bup-blo|, |bup-blo|)
    for j=1:5
        vlow= -abs(Uup(j) - Ulow(j));
        vup= abs(Uup(j) - Ulow(j));
        q(j,i)= vlow + rand*(vup-vlow);
    end
end



fname=strcat(output_folder,'/', 'iter_all',num2str(n_iter_all),'iter1_',num2str(n_iter),'-iter2_',num2str(n_iter2));

mkdir(fname)
  figure(1);
  hold on;
  mutation_ind=0;
  for i=1:n_iter
    for j=1:n_particle
        for d= 1:5
            rg=rand;
            rp=rand;
            % Update the particle's velocity
            vi= w*q(d,j)+ c1*rp*(r(d,j)-p(d,j))+ c2*rg*(g(d)-p(d,j));
             %Update the swarm's best known position: g : pi
             p(d,j) = p(d,j) + vi;
         end
         if (psoerror(logA,R,logR,M,p(1:5,j),nA,R01)< psoerror(logA,R,logR,M,r(1:5,j),nA,R01))
            r(1:5,j) = p(1:5,j);
         end
         if (psoerror(logA,R,logR,M,r(1:5,j),nA,R01)< psoerror(logA,R,logR,M,g,nA,R01))
            g = r(1:5,j);
         end
    end
    figure(2);
    plot(p(1,:),p(4,:),'.');
    xlim([Ulow(1) Uup(1)])
    ylim([Ulow(4) Uup(4)])
    pause(0.1);
    if (i==1)
        iter1(:,1)=p(2,:);
        iter1(:,2)=p(3,:);
    end
    if (i==n_iter/2)
        iter10(:,1)=p(2,:);
        iter10(:,2)=p(3,:);
    end
    if (i==n_iter)
        iter20(:,1)=p(2,:);
        iter20(:,2)=p(3,:);
    end
    %check position (here we check positions of the particles and if the 
    % position exceed max and min of the limits we replace it by mean
    for ch=1:n_particle
        for ch2=1:5
            if (p(ch2,ch) < Ulow(ch2))
                p(ch2,ch)=p(ch2,ch)+abs(Ulow(ch2)-Uup(ch2));
            end
            if (p(ch2,ch) > Uup(ch2))
                p(ch2,ch)=p(ch2,ch)-abs(Ulow(ch2)-Uup(ch2));
            end
        end  
    end
    %=======================================mutation=================================
    %=============================================================================
    if  (i==round(n_iter/2)) 
        for mui=1:5
            g(mui)= Ulow(mui) + rand*(Uup(mui) - Ulow(mui));   %random position for g position
            gmut=g;
        end
    end
    if (i==round(n_iter/2)+1 || i==round(n_iter/2)+2 || i==round(n_iter/2)+3)
        mutation_ind=1;
        g=gmut;
    else 
        mutation_ind=0;
    end
  end
  
  
  
  
  std_g = (psoerror(logA,M,R,logR,g,nA,R01));
  ii=0;
  for i=1:n_particle
      std_p = (psoerror(logA,M,R,logR,p(1:5,i),nA,R01));
      ss=ss+1;
      if std_p < 2*std_g
        ii=ii+1;
        finans(1:5,ii)= p(1:5,i);
      end 
  end
%   voronoi(finans(1,:),finans(2,:));
 
  
  
  for i=1:n_particle
    e(i)= (psoerror(logA,M,R,logR,p(1:5,i),nA,R01));
  end

thresh=0.02; %again for forcing a threshhold, doesnt do anything in this 
             %code as it will be calculated in dismin4 func

ind_show=1;
[t3,disind]= dismin4(n_particle,e,p,thresh,chkcond,n_ans,ind_show, logA);
  A=sort(disind);
  kdum=0;
  particle_idum2=0;
  motaghi_dum(1)=0;
  motaghi_dum(2)=0;
  for i=1:n_iter2
      i
    for j=1:n_particle
        for d= 1:5
            rp= rand;
            rg= rand;
            % Update the particle's velocity
            vi= w*q(d,j)+ c1*rp*(r(d,j)-p(d,j))+ c3*rg*(t3(disind(j,2),d)-p(d,j));
            q(d,j)= vi;
            %Update the particle's position: xi : xi + vi
            %if f(xi) < f(pi) then
            %Update the particle's best known position: pi : xi
            %if f(pi) < f(g) then
            %Update the swarm's best known position: g : pi
             p(d,j) = p(d,j) + vi;
        end
        t3d=t3';
        dum1=abs(psoerror(logA,R,logR,M,p(1:5,j),nA,R01)-psoerror(logA,R,logR,M,t3d(1:5,disind(j,2)),nA,R01));
        
        error_peri(j,i)=psoerror(logA,R,logR,M,p(1:5,j),nA,R01);
        dum2=abs(psoerror(logA,R,logR,M,r(1:5,j),nA,R01)-psoerror(logA,R,logR,M,t3d(1:5,disind(j,2)),nA,R01));
        if (dum1<dum2)
            r(1:5,j) = p(1:5,j);
        end
        if (psoerror(logA,R,logR,M,r(1:5,j),nA,R01)< psoerror(logA,R,logR,M,t3d(1:5,disind(j,2)),nA,R01))
            t3d(1:5,disind(j,2)) = r(1:5,j);
        end
    end
    figure(5);
    hold on
    scatter(t3(:,1),t3(:,2),30,'r','filled')
    hold on
    plot(p(1,:),p(2,:),'.');
    pause(0.5);
    hold off
    %==================================================
    if (chkstd_ind==1)      
        if (mod(i,n_update)==0)
            pdum=p'; 
            for pind=1:n_particle   
                pdum(pind,6)= psoerror(logA,R,logR,M,p(1:5,pind),nA,R01);
            end
            chkcond=2.88;
            thresh=0.4;
            %[pdum2,n_particle] = chkstd(n_particle,pdum,chkcond);
            ind_show=0;
            motaghi_dum(1)=motaghi_dum(2)+1;
            [t3,disind]= dismin4(n_particle,pdum(:,6),pdum(:,1:5)',thresh,chkcond,n_ans,ind_show, logA);
            motaghi_dum(2)=motaghi_dum(1)+n_ans-1;
            t3_show_m(motaghi_dum(1):motaghi_dum(2),1:6)=t3;
            t3_show_m(motaghi_dum(1):motaghi_dum(2),7)=t3(1,6)+thresh;
            t3_show_m(motaghi_dum(1):motaghi_dum(2),8)=i;
            A=sort(disind);
            p=pdum(:,1:5)';
        end
    end
    %========================================================
    %check position (here we check positions of the particles and if the 
    % position exceed max and min of the limits we replace it by mean
    for ch=1:n_particle
        for ch2=1:5
            if (p(ch2,ch) < Ulow(ch2))
                p(ch2,ch)=p(ch2,ch)+abs(Ulow(ch2)-Uup(ch2));
            end
            if (p(ch2,ch) > Uup(ch2))
                p(ch2,ch)=p(ch2,ch)-abs(Ulow(ch2)-Uup(ch2));
            end
        end  
    end
    
    merror_peri(i)=mean(error_peri(:,i));
    
    

    particle_idum1=particle_idum2+1;
    particle_idum2=i*n_particle;
    ipdum=0;
   for kp=particle_idum1:particle_idum2
        ipdum=ipdum+1;
        all_particle(kp,1:5)=p(1:5,ipdum);
        error_dum=psoerror(logA,R,logR,M,p(1:5,ipdum),nA,R01);
        all_particle(kp,6)=error_dum;
        all_particle(kp,7)=n_iter;
        all_particle(kp,8)=i;
   end
   %=================================================
  end
  
  
  [dum1 dum2]=size(t3d);
  for i=1:dum2      
    [valout,e] = errorplot(logA,R,logR,M,t3d(1:5,i),nA,R01);
    ef(i)=e;
    maxdis=max(valout(:,2));
    mindis=min(valout(:,2));
    difdis=20;
    pcurdis=0.0;
    curdis=difdis;
    jdum=0;
    while (curdis < maxdis)
        jdum=jdum+1;
        idum=0;
        dum=0;
        for j=1:nA
            if (valout(j,2) <= curdis && valout(j,2) > pcurdis)
                idum=idum+1;
                o1(idum,i)=valout(j,1);
                dum=(valout(j,1)*valout(j,1))+dum;
            end
        end
        out_1(jdum,i)=sqrt(dum/idum);
        out_dis(jdum,1)=curdis;
        pcurdis=curdis;
        curdis=curdis+difdis; 
    end
    figure
    subplot(2,1,1)
    scatter(valout(:,2),valout(:,1),'.')
    subplot(2,1,2)
    scatter(out_dis(:,1),out_1(:,i),'.') 
    
    saveas(gcf,strcat(fname,'/res_dis_',num2str(i),'.png'))
  end
  

  [ist3 jst3]=size(t3);
   spind=round(ist3/3);
   if (ist3 > spind * 3)
       spind=spind+1;
   end
   spind2=3;
   idum=0;
   FigH = figure('Position', get(0, 'Screensize'));

  for i=1:ist3
    idum=idum+1;
    [yobs,ysyn] = psoerror_for_obs_synth(logA,R,logR,M,t3d(1:5,i),nA,R01);
    subplot(spind,spind2,idum)
    scatter(R,ysyn)
    hold on
    scatter(R,yobs,'.','r')
    ylim([-11 0])
  end
  F    = getframe(FigH);
  imwrite(F.cdata,strcat(fname,'/res_out_','.png'), 'png')
%   saveas(gcf,strcat(fname,'/res_out_',fname,'.png'))
  %%
  [nt3di mt3di]=size(t3d);
  figure
  for i=1:mt3di
    hold on
    [yobs,ysyn] = psoerror_for_obs_synth(logA,R,logR,M,t3d(1:5,i),nA,R01);
    scatter(R,ysyn,'.')
    ylim([-11 0])
  end 
  legend('show');
  saveas(gcf,strcat(fname,'/outpputs_','.png'))
  
  
  %creating some output for python plotter
  %python code need t3_ls_merge, all_particle_for_save
  
  
  output_name=sprintf('%s_%d_%d%s','t3out',n_particle, n_iter,'.dat');
  save(output_name,'t3','-ascii')
  t3_ls_merge=t3;
  t3_ls_merge(ist3+1,:)=pos_ls(1:6);
  save('pso_ls_merge.dat','t3_ls_merge', '-ascii');
  
  hold off
  merror_peri=merror_peri';
  plot(merror_peri)
  saveas(gcf,strcat(fname,'/error_mean_','.fig'))
 all_particle_for_save = all_particle(:, 1:6);
 if (synth_ind == 1)
    fl_name = strcat(output_folder, '/', 'all_particle_for_save_syn.dat');
    save(fl_name, 'all_particle_for_save', '-ascii');
 else
     fl_name = strcat(output_folder, '/', 'all_particle_for_save.dat');
     save(fl_name, 'all_particle_for_save', '-ascii');
 end
 fl_name = strcat(output_folder, '/', 't3_ls_merge.dat');
 save(fl_name, 't3_ls_merge', '-ascii');
 
close all



%termina output the PSO solutions and the last one is the least square
%solution
t3_ls_merge