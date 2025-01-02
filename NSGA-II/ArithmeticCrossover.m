function [popc1, popc2]=ArithmeticCrossover(p1,p2, GridNumber, Size)
    global NFE;
    NFE = NFE+1;
    
    a1 = randi(GridNumber);
    AFlag = true;
    while a1>0 && AFlag
        [popc1, AFlag] = Add(p1, GridNumber, Size);
        a1 = a1-1;
    end

    a2 = randi(GridNumber);
    RFlag = true;
    while a2>0 && RFlag
        [popc2, RFlag] = Remove(p2, GridNumber, Size);
        a2 = a2-1;
    end
    
end