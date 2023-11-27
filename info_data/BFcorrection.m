% [ ind, ToneDriven, bf, field]
Correction = [ 162, NaN, 3.20,NaN;
    199, NaN, NaN, 2;
    217, 2,   NaN, NaN; % 2 for ToneDriven (not driven, but come up as significant by ttest)
    219, NaN, NaN, 2;
    221, 2, NaN, NaN;
    228, 2, NaN, NaN;
    234, 2, NaN, NaN;
    510, -1,NaN, NaN; % 512- 522 (ADF, 963-1) ???
    652, -1, NaN, NaN;
    661, -1, NaN, NaN;
    662, -1, NaN, NaN;
    663, -1, NaN, NaN;
    682, 2, NaN, NaN;
    683, -1, NaN, NaN;
    684, -1, NaN, NaN;
    685, -1, NaN, NaN;
    686, -1, NaN, NaN;
    687, -1, NaN, NaN;
    691, -1, NaN, NaN;
    696, -1, NaN, NaN;
      5,  2, NaN, NaN; % animal 832
     22, -1, NaN, NaN;
     25, -1, NaN, NaN;
     41,  2, NaN, NaN; % 833
     43,  2, NaN, NaN;
     44,  2, NaN, NaN;
     45,  2, NaN, NaN;
     48,  2, NaN, NaN;
     53,  2, NaN, NaN;
     97, -1, NaN, NaN;
    110, -1, NaN, NaN;
    112, -1, NaN, NaN; 
    122, -1, NaN, NaN; %848
    125,  2, NaN, NaN;
    127, -1, NaN, NaN;
    146,  2, NaN, NaN;
    151,  2, NaN, NaN;
    152,  2, NaN, NaN;
    164, -1, NaN, NaN;
    183,  2, NaN, NaN;
    202, -1, NaN, NaN; % 943
    209,  2, NaN, NaN;
    217, -1, NaN, NaN;
    234,  2, NaN, NaN;
    236, -1, NaN, NaN;
    240, -1, NaN, NaN;
    251, -1, NaN, NaN;
    252, -1, NaN, NaN;
    254, -1, NaN, NaN;
    299, -1, NaN, NaN;
    308, -1, NaN, NaN;
    315, -1, NaN, NaN;
    318, -1, NaN, NaN;
    321, -1, NaN, NaN;
    323,  2, NaN, NaN;
    329,  2, NaN, NaN;
    408,  2, NaN, NaN;
    438, -1, NaN, NaN;
    462,  2, NaN, NaN;
    510, -1 ,NaN, NaN;
    564, -1, NaN, NaN; % 963
    597, -1, NaN, NaN;
    607,  2, NaN, NaN;
    608,  2, NaN, NaN;    
    618,  2, NaN, NaN;
    620,  2, NaN, NaN;
    624,  2, NaN, NaN;
    632, -1, NaN, NaN;
    647, -1, NaN, NaN;
    652, -1 ,NaN, NaN;
    661, -1, NaN, NaN;
    662, -1, NaN, NaN;
    683, -1, NaN, NaN;
    684, -1, NaN, NaN;
    686, -1 ,NaN, NaN;
    687, -1, NaN, NaN;
    691, -1, NaN, NaN;
    692, -1, NaN, NaN;
    696, -1, NaN, NaN;
    697, -1, NaN, NaN;
    720,  2, NaN, NaN;
    722,  2, NaN, NaN;
    752, -1, NaN, NaN;
    766, -1, NaN, NaN;
    799, -1 ,NaN, NaN;
    802, -1, NaN, NaN;
    803,  2, NaN, NaN;
    804,  2, NaN, NaN;
    805,  2, NaN, NaN;
    806,  2, NaN, NaN;
    811,  2, NaN, NaN;
    814,  2, NaN, NaN;
    815,  2, NaN, NaN;
    816,  2, NaN, NaN;
    817,  2, NaN, NaN;
    826,  2, NaN, NaN;
    ];

for k=1:length(Correction)
    if ~isnan(Correction(k,2))
        trained_metrics(Correction(k,1)).ToneDriven = Correction(k,2);
    end
    if ~isnan(Correction(k,3))
        trained_metrics(Correction(k,1)).BF = Correction(k,3);
    end
    if ~isnan(Correction(k,4))
        trained_metrics(Correction(k,1)).field = Correction(k,4);
    end
end

