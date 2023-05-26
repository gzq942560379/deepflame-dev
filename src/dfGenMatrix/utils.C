#include "GenFvMatrix.H"


namespace Foam{

#define ERROR_TOLERANCE 1e-6

#define check_error(error, name) { \
    if(error > ERROR_TOLERANCE){ \
        SeriousError << name << " error too large : " << error << endl; \
        MPI_Abort(PstreamGlobals::MPI_COMM_FOAM, -1); \
    } \
}


void check_field_error(Field<scalar>& a, Field<scalar>& b, const word& name){
    if(a.size() != b.size()){
        SeriousError << "In check_field_error, two field have different size !!!" << endl;
        MPI_Abort(PstreamGlobals::MPI_COMM_FOAM, -1);
    }
    double max_absulte_error = 0.;
    double max_relative_error = 0.;
    for(label i = 0; i < a.size(); ++i){
        double absulte_error = std::abs(a[i] - b[i]);
        double relative_error = absulte_error / std::abs(a[i]);
        max_absulte_error = std::max(max_absulte_error, absulte_error);
        max_relative_error = std::max(max_relative_error, relative_error);
    }
    if(max_relative_error > 1e-12){
        Info << name << " error : " << endl;
        Info << "max_absulte_error : " << max_absulte_error << endl;
        Info << "max_relative_error : " << max_relative_error << endl;
        MPI_Abort(PstreamGlobals::MPI_COMM_FOAM, -1);
    }
}

void check_fvmatrix_equal(fvScalarMatrix& a,fvScalarMatrix& b){
    if(a.source().begin() == b.source().begin()){
        SeriousError << "a and b are the same matrix." << endl;
        MPI_Abort(PstreamGlobals::MPI_COMM_FOAM, -1);
    } 
    check_field_error(a.source(),b.source(), "source");
    check_field_error(a.diag(),b.diag(), "diag");
    check_field_error(a.lower(),b.lower(), "lower");
    check_field_error(a.upper(),b.upper(), "upper");
    for(label patchi = 0; patchi < a.internalCoeffs().size(); ++patchi)
    {
        check_field_error(a.internalCoeffs()[patchi],b.internalCoeffs()[patchi], "internalCoeffs_" + std::to_string(patchi));
        check_field_error(a.boundaryCoeffs()[patchi],b.boundaryCoeffs()[patchi], "boundaryCoeffs_" + std::to_string(patchi));
    }
}

}
