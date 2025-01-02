function popc1=ArithmeticAdd(p1, GridNumber, Size)
    global NFE;
    NFE = NFE+1;
    
    a1 = randi(GridNumber);
    AFlag = true;
    popc1 = p1;
    disp(a1)
    while a1>0 && AFlag
        [popc1, AFlag] = Add(popc1, GridNumber, Size);
        disp(AFlag)
        a1 = a1-1;
    end
    
end