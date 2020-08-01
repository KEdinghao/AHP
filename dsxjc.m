% m��׼��
% n������
% A1���߾���
% A2Ȩ�ؾ���
% a��������(m==size(a))
% 1Ч����(Խ��Խ��) 2�ɱ���(ԽСԽ��) 3�̶���(Խ�ӽ��̶�ֵԽ��) 4ƫ����(ԽԶ��̶�ֵԽ��) 5�̶������� 6ƫ��������
% c1�̶�ֵ(������Ϊ�����̶�ֵ) c2ƫ��ֵ C1�̶����� C2ƫ������

%% ���ݵ���
x=2000:10:6000;
x_tuition=[3350 3895 4224 4119 4785 5070 5185];
y_cost=[15974 15445 15120 14963 14627 14295 14008];
y_gdp=[43648 47173 50237 54139 60014 66006 70892];
y_country=[0.355264651	0.371344946	0.369822356	0.333160781	0.356228224	0.346145968	0.323657928];
y_city=[0.126572713	0.135037218	0.135407183	0.113255172	0.131469769	0.129169342	0.12240899];
city_proportion=[0.605999786 0.59580186	0.585196535	0.573496973	0.560998676	0.547703645	0.537296431];
city_proportion=city_proportion(:,end:-1:1);%������뷴�ˣ�����һ��
country_proportion=1-city_proportion;

%% �������
%cost
y3_cost=polyfit(x_tuition,y_cost,3);
yi3_cost=polyval(y3_cost,x);
%gdp
y3_gdp=polyfit(x_tuition,y_gdp,2);
yi3_gdp=polyval(y3_gdp,x);
%major
y_major(find(x<=3400))=2;
y_major(find(3400<x&x<=3550))=4;
y_major(find(3650<x&x<=4100))=3;
y_major(find(x>=4100))=1;
%press
y_press=y_country.*country_proportion+y_city.*city_proportion;
y3_press=polyfit(x_tuition,y_press,4);
yi3_press=polyval(y3_press,x);

%% ԭʼ���ݾ���
A_initial=[yi3_cost',yi3_gdp',y_major',yi3_press'];

%%   ���߾���_�����Ծ��߷�
A1=A_initial;
a=[2,1,1,2];
c1=6;
R=ones(size(A1));  % size(A1,1)����
[n,m] = size(A1);
% �����任�ٶ�: ���Ե���Ҫ��������ֵ���Ա仯.
for i=1:m %��
    ai=a(i);
    switch ai
        case 1
            for j=1:n %��
                if R(j,i)~=0
                    R(j,i) = A1(j,i)/max(A1(:,i));
                else
                    R(j,i) = 0;
                end
            end
        case 2
            for j=1:n
                if R(j,i)~=0
                    R(j,i) = min(A1(:,i))/A1(j,i);
                else
                    R(j,i) = 0;
                end
            end
        case 3
            for j=1:n
                if R(j,i)~=0
                    R(j,i) = 1 - (A1(j,i)-c1)/max(abs(A1(:,i)-c1));
                else
                    R(j,i) = 0;
                end
            end
        case 4
            for j=1:n
                if R(j,i)~=0
                    R(j,i) = abs(A1(j,i)-c2) - min(abs(A1(:,i)-c2))/(max(abs(A1(:,i)-c2))-min(abs(A1(:,j)-c2)));
                else
                    R(j,i) = 0;
                end
            end
    end
end

%%   Ȩ�ؾ���_��η�����(AHP)
A2=[1 5 1/3 1/5
    1/3 1 1/3 1/5
    2 5 1 1/4
    4 5 2 1];
%һ���Լ����Ȩ��������
n2=size(A2,1);
[v,d]=eig(A2);                    %������ֵ������������V�������������D�������ֵ
Max_eig =max(max(d));                         %ȡD���������ֵ������r=max(max(D))    
CI=(Max_eig-n2)/(n2-1);
if abs(CI)<0.000001
    CI=0;
end
RI=[0 0 0.58 0.90 1.12 1.24 1.32 1.41 1.45 1.49 1.52 1.54 1.56 1.58 1.59];%���ƽ�����һ����ָ��
CR=CI/RI(n2);
if  CR<0.10
    CR_Result='��ΪCR < 0.10�����Ը��жϾ���A��һ���Կ��Խ���!';
   else
    CR_Result='ע�⣺CR >= 0.10����˸��жϾ���A��Ҫ�����޸�!';  
end

%% Ȩ��������
%����1������ƽ������Ȩ��CI
sum_A=sum(A2);                     %�Ծ�������������
SUM_A=repmat(sum_A,n2,1);           %��sum_A����һ�����壬�ظ�n*1��
stand_A=A2./SUM_A;                 %A��SUM_A�����Ӧ������������жϾ������й�һ��
w_am=sum(stand_A,2)/n2;

%����2������ƽ������Ȩ��
Prduct_A = prod(A2,2);              %��A��Ԫ�ذ�������˵õ�һ���µ�������
% prod������sum�������ƣ�һ�����ڳˣ�һ�����ڼ�  dim = 2 ά������
Prduct_n_A = Prduct_A .^ (1/n2);     % �����ÿ��Ԫ�ؽ��г˷������������ǿ�n�η����������ǵȼ���1/n�η�
w_gm=Prduct_n_A ./sum(Prduct_n_A);

%����3������ֵ����Ȩ��
d=round(d);
[r,c] = find(d == Max_eig  , 1);           % �ҵ�D�е�һ�����������ֵ��ȵ�Ԫ�ص�λ�ã���¼�����к��С�
w_em=v(:,c)./sum(v(:,c));           %������������������й�һ�����ɵõ����ǵ�Ȩ��

%% �ۺϷ���(ȡ����ƽ������Ȩ��)
%����1���򵥼�Ȩ�ͷ�
for j=1:n   
    v_saw(j)=sum(R(j,:)*w_am);
end
%����2����Ȩ����
for j=1:n   
    v_wp(j)=prod(power(R(j,:),w_gm'),2);%prod(A,2)�гɼ�
end

%% Ѱ��λ�ã�����Ӧѧ��
x_position_saw = find(v_saw==max(v_saw));
x_position_wp = find(v_wp==max(v_wp));

%% ������
disp('���������ݱ��棺');
% disp(table(R));
disp(['һ����ָ��:' num2str(CI)]);
disp(['һ���Ա���:' num2str(CR)]);
disp(['һ���Լ�����:' CR_Result]);
% disp(['����ֵ:' num2str(Max_eig )]);
 disp('����ƽ������Ȩ�صĽ��Ϊ��');
 disp(w_am);
% disp('����ƽ������Ȩ�صĽ��Ϊ��');
% disp(w_gm);
% disp('����ֵ����Ȩ�صĽ��Ϊ��');
% disp(w_em);
% disp('�򵥼�Ȩ�ͷ�������˳����Ϊ');
% disp(v_saw);
% disp('��Ȩ����������˳����Ϊ');
% disp(v_wp);
disp('���ʵ�ѧ��sawΪ��');
disp(x(x_position_saw));
disp('���ʵ�ѧ��wpΪ��');
disp(x(x_position_wp));
subplot(2,2,1)
plot(x,yi3_gdp)
title('gdp')
subplot(2,2,2)
plot(x,yi3_cost)
title('cost')
subplot(2,2,3)
plot(x,y_major)
title('major')
subplot(2,2,4)
plot(x,yi3_press)
title('press')
figure 
subplot(1,2,1)
plot(x,v_saw)
title('SAW�㷨�÷�')
subplot(1,2,2)
plot(x,v_wp)
title('WP�㷨�÷�')