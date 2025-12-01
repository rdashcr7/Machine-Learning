clear all; clc

[num1,txt1,raw1] = xlsread('Z-scale.xlsx'); 

[num2,txt2,raw2] = xlsread("700 modified KP sequences 13 Oct.xlsx");

peptide_count = size(txt2); 
TPC = peptide_count(1,1);

for i = 2:TPC
    Peptide = txt2(i,1);
    AA = char(Peptide);
    L(i) = strlength(Peptide);
    PL = L(i); % peptide length
    for j=1:PL
        for k=2:21
            if AA(j:j) == string(txt1(k,1))
                P1(i-1,j) = num1(k-1,1);
                P2(i-1,j) = num1(k-1,2);
                P3(i-1,j) = num1(k-1,3);
            end
        end
    end
end

P = [P1,P2,P3];
