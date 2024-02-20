Info<< "Constructing fields for refined mesh\n" << endl;
Info<< "Reading data from file\n" << endl;
const string filePath = CanteraTorchProperties.subDict("RefineSettings").lookupOrDefault("filePath", string(""));
const label girdSize = CanteraTorchProperties.subDict("RefineSettings").lookupOrDefault("girdSize", 0);

label row = Y.size() + 7;
label col = girdSize;
double* filedata = new double[row * col];

int mpirank, mpisize;
MPI_Initialized(&flag_mpi_init);

if(flag_mpi_init){
    MPI_Comm_rank(PstreamGlobals::MPI_COMM_FOAM, &mpirank);
    MPI_Comm_size(PstreamGlobals::MPI_COMM_FOAM, &mpisize);
}

if(mpirank == 0 || !flag_mpi_init){
    std::ifstream infile;
    std::string str;
    infile.open(filePath);
    for(label r = 0; r < row; ++r){
        std::getline(infile, str);
        std::stringstream ss(str);
        string tmp;
        for(label j = 0; j < col; ++j)
        {
            ss >> tmp;
            filedata[r * col + j] = std::stod(tmp); 
        }
        // Info <<"Done Reading Row #" << r << endl;
    }
}

if(flag_mpi_init){
    MPI_Bcast(filedata, row * col, MPI_DOUBLE, 0, PstreamGlobals::MPI_COMM_FOAM);
}

const double* filedata_x = filedata;
const double* filedata_T = filedata + col;
const double* filedata_ha = filedata + 2 * col;
const double* filedata_psi = filedata + 3*col;
const double* filedata_rho = filedata + 4 * col;
const double* filedata_mu = filedata + 5 * col;
const double* filedata_alpha = filedata + 6 * col;
const double* filedata_Y = filedata + 7 * col;

scalar x, y, z;
label index;
forAll(mesh.C(), celli)
{
    x = mesh.C()[celli][0];
    y = mesh.C()[celli][1];
    z = mesh.C()[celli][2];

    // set scalar fields
    index = round((x/0.00003) / filedata_x[1]); 
    T[celli] = filedata_T[index]; 
    rho[celli] = filedata_rho[index];
}
T.correctBoundaryConditions();
rho.correctBoundaryConditions();


forAll(mesh.C(), celli)
{
    x = mesh.C()[celli][0];
    y = mesh.C()[celli][1];
    z = mesh.C()[celli][2];
    // set filed U
    U[celli] = vector(4*std::sin(x/0.00003)*std::cos(y/0.00003)*std::cos(z/0.00003),-4*std::cos(x/0.00003)*std::sin(y/0.00003)*std::cos(z/0.00003),0.0); 
}

U.correctBoundaryConditions();

phi = linearInterpolate(rho*U) & mesh.Sf();

forAll(mesh.C(), celli)
{
    x = mesh.C()[celli][0];
    y = mesh.C()[celli][1];
    z = mesh.C()[celli][2];

    index = round((x/0.00003) / filedata_x[1]);

    for (int speciI = 0; speciI < Y.size(); speciI++)
    {
        Y[speciI][celli] = filedata_Y[speciI * col + index];
    }
}
for (int speciI = 0; speciI < Y.size(); speciI++)
{
    Y[speciI].correctBoundaryConditions();
}


forAll(mesh.C(), celli)
{
    x = mesh.C()[celli][0];
    y = mesh.C()[celli][1];
    z = mesh.C()[celli][2];

    // set scalar fields
    index = round((x/0.00003) / filedata_x[1]);
    thermo.he()[celli] = filedata_ha[index];
}
thermo.he().correctBoundaryConditions();

chemistry->correctThermo();
rho = thermo.rho();

initialized = true;