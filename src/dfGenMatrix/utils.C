#include "GenFvMatrix.H"


namespace Foam{

#define RELATIVE_ERROR_TOLERANCE 1e-9
#define ABSULTE_ERROR_TOLERANCE 1e-18

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
        Info << name << " error : " << endl;
        Info << "max_absulte_error : " << max_absulte_error << endl;
        Info << "max_relative_error : " << max_relative_error << endl;
    if(max_relative_error > RELATIVE_ERROR_TOLERANCE && max_absulte_error > ABSULTE_ERROR_TOLERANCE){
        MPI_Abort(PstreamGlobals::MPI_COMM_FOAM, -1);
    }
}

void check_field_equal(Field<scalar>& a, Field<scalar>& b){
    check_field_error(a, b, "field");
    return;
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

void check_fvmatrix_equal(fvVectorMatrix& a,fvVectorMatrix& b){
    if(a.source().begin() == b.source().begin()){
        SeriousError << "a and b are the same matrix." << endl;
        MPI_Abort(PstreamGlobals::MPI_COMM_FOAM, -1);
    } 

    for (direction cmpt=0; cmpt<vector::nComponents; ++cmpt)
    {
        check_field_error(a.source().component(cmpt).ref(),b.source().component(cmpt).ref(), "source_" + std::to_string(cmpt));
    }

    check_field_error(a.diag(),b.diag(), "diag");
    check_field_error(a.lower(),b.lower(), "lower");
    check_field_error(a.upper(),b.upper(), "upper");

    for (direction cmpt=0; cmpt<vector::nComponents; ++cmpt)
    {
        forAll(a.internalCoeffs().component(cmpt)(), patchi)
        {
            check_field_error(a.internalCoeffs().component(cmpt).ref()[patchi],b.internalCoeffs().component(cmpt).ref()[patchi], "internalCoeffs_" + std::to_string(cmpt) + "_" + std::to_string(patchi));
            check_field_error(a.boundaryCoeffs().component(cmpt).ref()[patchi],b.boundaryCoeffs().component(cmpt).ref()[patchi], "boundaryCoeffs_" + std::to_string(cmpt) + "_" + std::to_string(patchi));
        }
    }
}

}
