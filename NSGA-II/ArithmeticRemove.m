function popc2=ArithmeticRemove(p2, GridNumber, Size)
    global NFE;
    NFE = NFE+1;

    a2 = randi(GridNumber);
    RFlag = true;
    popc2 = p2;
    disp(a2)
    while a2>0 && RFlag
        [popc2, RFlag] = Remove(popc2, GridNumber, Size);
        disp(RFlag)
        a2 = a2-1;
    end
    
end