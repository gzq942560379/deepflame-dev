Info<< "Constructing fields for refined mesh\n" << endl;
Info<< "Reading data from file\n" << endl;
const string filePath = CanteraTorchProperties.subDict("RefineSettings").lookupOrDefault("filePath", string(""));
const label girdSize = CanteraTorchProperties.subDict("RefineSettings").lookupOrDefault("girdSize", 0);

std::vector<double *> filedata;
std::ifstream infile;
std::string str;
infile.open(filePath);
while (std::getline(infile, str))
{
    double * data = new double[girdSize]{0.0};
    std::stringstream sumstr(str);
    string out;
    int i = 0;
    while (sumstr >> out)
    {
        double a = std::stod(out); 
        data[i] = a;
        i++;
    }
    filedata.push_back(data);
}

scalar x, y, z;
label index;
forAll(mesh.C(), celli)
{
    x = mesh.C()[celli][0];
    y = mesh.C()[celli][1];
    z = mesh.C()[celli][2];

    // set scalar fields
    index = round((x/0.001) / filedata[0][1]);
    
    T[celli] = filedata[1][index]; 
}
T.correctBoundaryConditions();

forAll(mesh.C(), celli)
{
    // vector c = mesh.C()[celli];
    x = mesh.C()[celli][0];
    y = mesh.C()[celli][1];
    z = mesh.C()[celli][2];

    // set filed U
    U[celli] = vector(4*std::sin(x/0.001)*std::cos(y/0.001)*std::cos(z/0.001),-4*std::cos(x/0.001)*std::sin(y/0.001)*std::cos(z/0.001),0.0); 
}
U.correctBoundaryConditions();

forAll(mesh.C(), celli)
{
    x = mesh.C()[celli][0];
    y = mesh.C()[celli][1];
    z = mesh.C()[celli][2];

    index = round((x/0.001) / filedata[0][1]);

    for (int speciI = 0; speciI < Y.size(); speciI++)
    {
        Y[speciI][celli] = filedata[2+speciI][index];
    }
}
for (int speciI = 0; speciI < Y.size(); speciI++)
{
    Y[speciI].correctBoundaryConditions();
}

chemistry->correctEnthalpy();
chemistry->correctThermo();
rho = p * psi;
rho.correctBoundaryConditions();

initialized = true;