
if exist('averagepooling')~=3
    mex averagepooling.cpp COMPFLAGS="/openmp $COMPFLAGS" CXXFLAGS="$CFLAGS -fPIC -fopenmp" LDFLAGS="$LDFLAGS -fopenmp" -largeArrayDims 
end;
if exist('maxpooling')~=3
    mex maxpooling.cpp COMPFLAGS="/openmp $COMPFLAGS" CXXFLAGS="$CFLAGS -fPIC -fopenmp" LDFLAGS="$LDFLAGS -fopenmp" -largeArrayDims 
end;
if exist('stochasticpooling')~=3
    mex stochasticpooling.cpp COMPFLAGS="/openmp $COMPFLAGS" CXXFLAGS="$CFLAGS -fPIC -fopenmp" LDFLAGS="$LDFLAGS -fopenmp" -largeArrayDims 
end;