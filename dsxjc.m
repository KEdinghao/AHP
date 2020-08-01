% m个准则
% n个方案
% A1决策矩阵
% A2权重矩阵
% a数据类型(m==size(a))
% 1效益型(越大越好) 2成本型(越小越好) 3固定型(越接近固定值越好) 4偏离型(越远离固定值越好) 5固定区间型 6偏离区间型
% c1固定值(本代码为单个固定值) c2偏离值 C1固定区间 C2偏离区间

%% 数据导入
x=2000:10:6000;
x_tuition=[3350 3895 4224 4119 4785 5070 5185];
y_cost=[15974 15445 15120 14963 14627 14295 14008];
y_gdp=[43648 47173 50237 54139 60014 66006 70892];
y_country=[0.355264651	0.371344946	0.369822356	0.333160781	0.356228224	0.346145968	0.323657928];
y_city=[0.126572713	0.135037218	0.135407183	0.113255172	0.131469769	0.129169342	0.12240899];
city_proportion=[0.605999786 0.59580186	0.585196535	0.573496973	0.560998676	0.547703645	0.537296431];
city_proportion=city_proportion(:,end:-1:1);%年份输入反了，反序一下
country_proportion=1-city_proportion;

%% 数据拟合
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

%% 原始数据矩阵
A_initial=[yi3_cost',yi3_gdp',y_major',yi3_press'];

%%   决策矩阵_多属性决策法
A1=A_initial;
a=[2,1,1,2];
c1=6;
R=ones(size(A1));  % size(A1,1)行数
[n,m] = size(A1);
% 比例变换假定: 属性的重要性随属性值线性变化.
for i=1:m %列
    ai=a(i);
    switch ai
        case 1
            for j=1:n %行
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

%%   权重矩阵_层次分析法(AHP)
A2=[1 5 1/3 1/5
    1/3 1 1/3 1/5
    2 5 1 1/4
    4 5 2 1];
%一致性检验和权向量计算
n2=size(A2,1);
[v,d]=eig(A2);                    %求特征值和特征向量，V存放特征向量，D存放特征值
Max_eig =max(max(d));                         %取D中最大特征值，或者r=max(max(D))    
CI=(Max_eig-n2)/(n2-1);
if abs(CI)<0.000001
    CI=0;
end
RI=[0 0 0.58 0.90 1.12 1.24 1.32 1.41 1.45 1.49 1.52 1.54 1.56 1.58 1.59];%存放平均随机一致性指标
CR=CI/RI(n2);
if  CR<0.10
    CR_Result='因为CR < 0.10，所以该判断矩阵A的一致性可以接受!';
   else
    CR_Result='注意：CR >= 0.10，因此该判断矩阵A需要进行修改!';  
end

%% 权向量计算
%方法1：算术平均法求权重CI
sum_A=sum(A2);                     %对矩阵所在列作和
SUM_A=repmat(sum_A,n2,1);           %将sum_A看做一个整体，重复n*1块
stand_A=A2./SUM_A;                 %A和SUM_A矩阵对应行相除；即将判断矩阵按照列归一化
w_am=sum(stand_A,2)/n2;

%方法2：几何平均法求权重
Prduct_A = prod(A2,2);              %将A的元素按照行相乘得到一个新的列向量
% prod函数和sum函数类似，一个用于乘，一个用于加  dim = 2 维度是行
Prduct_n_A = Prduct_A .^ (1/n2);     % 这里对每个元素进行乘方操作，这里是开n次方，所以我们等价求1/n次方
w_gm=Prduct_n_A ./sum(Prduct_n_A);

%方法3：特征值法求权重
d=round(d);
[r,c] = find(d == Max_eig  , 1);           % 找到D中第一个与最大特征值相等的元素的位置，记录它的行和列。
w_em=v(:,c)./sum(v(:,c));           %对求出的特征向量进行归一化即可得到我们的权重

%% 综合方法(取算术平均法求权重)
%方法1：简单加权和法
for j=1:n   
    v_saw(j)=sum(R(j,:)*w_am);
end
%方法2：加权积法
for j=1:n   
    v_wp(j)=prod(power(R(j,:),w_gm'),2);%prod(A,2)行成绩
end

%% 寻找位置，并对应学费
x_position_saw = find(v_saw==max(v_saw));
x_position_wp = find(v_wp==max(v_wp));

%% 结果输出
disp('处理结果数据报告：');
% disp(table(R));
disp(['一致性指标:' num2str(CI)]);
disp(['一致性比例:' num2str(CR)]);
disp(['一致性检验结果:' CR_Result]);
% disp(['特征值:' num2str(Max_eig )]);
 disp('算术平均法求权重的结果为：');
 disp(w_am);
% disp('几何平均法求权重的结果为：');
% disp(w_gm);
% disp('特征值法求权重的结果为：');
% disp(w_em);
% disp('简单加权和法求优劣顺序结果为');
% disp(v_saw);
% disp('加权积法求优劣顺序结果为');
% disp(v_wp);
disp('合适的学费saw为：');
disp(x(x_position_saw));
disp('合适的学费wp为：');
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
title('SAW算法得分')
subplot(1,2,2)
plot(x,v_wp)
title('WP算法得分')