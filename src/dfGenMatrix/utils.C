#include "GenFvMatrix.H"


namespace Foam{

#define RELATIVE_ERROR_TOLERANCE 1e-12
#define ABSULTE_ERROR_TOLERANCE 1e-12
#define ZERO_TOLERANCE 1e-12

void check_field_error(const Field<scalar>& a, const Field<scalar>& b, const word& name){
    if(a.size() != b.size()){
        SeriousError << "In check_field_error, two field have different size !!!" << endl;
        MPI_Abort(PstreamGlobals::MPI_COMM_FOAM, -1);
    }
    // double max_absulte_error = 0.;
    bool check_faild = false;
    double max_relative_error = 0.;
    double aa, bb, max_absulte_error;
    for(label i = 0; i < a.size(); ++i){
        double absulte_error = std::abs(a[i] - b[i]);
        double relative_error = std::abs(a[i]) < ZERO_TOLERANCE ? absulte_error : absulte_error / std::abs(a[i]);
        if(absulte_error > ABSULTE_ERROR_TOLERANCE && relative_error > RELATIVE_ERROR_TOLERANCE){
            Info << name << " error : " << endl;
            Info << "a[i] : " << a[i] << endl;
            Info << "b[i] : " << b[i] << endl;
            Info << "absulte_error : " << absulte_error << endl;
            Info << "relative_error : " << relative_error << endl;
            check_faild = true;
        }
        if(relative_error > max_relative_error){
            aa = a[i];
            bb = b[i];
            max_absulte_error = absulte_error;
        }
    }
    if(check_faild){
        MPI_Abort(PstreamGlobals::MPI_COMM_FOAM, -1);
    }else{
        Info << "error check : " << name << "-------------------------------------" << endl;
        Info << "aa : " << aa << endl;
        Info << "bb : " << bb << endl;
        Info << "max_absulte_error : " << max_absulte_error << endl;
        Info << "max_relative_error : " << max_relative_error << endl;
        Info << "---------------------------------------------------------" << endl;
    }
    // if(max_relative_error > RELATIVE_ERROR_TOLERANCE && max_absulte_error > ABSULTE_ERROR_TOLERANCE){
    // if(max_relative_error > RELATIVE_ERROR_TOLERANCE){
    //     Info << name << " error : " << endl;
    //     Info << "max_relative_error : " << max_relative_error << endl;
    //     MPI_Abort(PstreamGlobals::MPI_COMM_FOAM, -1);
    // }
}

void check_field_equal(const Field<scalar>& a, const Field<scalar>& b){
    check_field_error(a, b, "field");
    return;
}

void check_field_boundary_equal(const volScalarField& a, const volScalarField& b){
    check_field_error(a, b, "field");
    forAll(a.boundaryField(), patchi)
    {
        check_field_error(a.boundaryField()[patchi], b.boundaryField()[patchi], "boundaryField_" + std::to_string(patchi));
    }
}

void check_field_boundary_equal(const volVectorField& a, const volVectorField& b){
    for (direction cmpt=0; cmpt<vector::nComponents; ++cmpt)
    {
        check_field_error(a.component(cmpt).ref(),b.component(cmpt).ref(), "internal_" + std::to_string(cmpt));
    }
    forAll(a.boundaryField(), patchi)
    {
        for (direction cmpt=0; cmpt<vector::nComponents; ++cmpt)
        {
            check_field_error(a.boundaryField()[patchi].component(cmpt).ref(),b.boundaryField()[patchi].component(cmpt).ref(), "boundary_" + std::to_string(cmpt));
        }
    }
}

void check_field_boundary_equal(const surfaceScalarField& a, const surfaceScalarField& b){
    check_field_error(a, b, "field");
    forAll(a.boundaryField(), patchi)
    {
        check_field_error(a.boundaryField()[patchi], b.boundaryField()[patchi], "boundaryField_" + std::to_string(patchi));
    }
}

void check_fvmatrix_equal(const fvScalarMatrix& a,const fvScalarMatrix& b){
    if(a.source().begin() == b.source().begin()){
        SeriousError << "a and b are the same matrix." << endl;
        MPI_Abort(PstreamGlobals::MPI_COMM_FOAM, -1);
    } 

    check_field_error(a.source(),b.source(), "source");
    check_field_error(a.diag(),b.diag(), "diag");
    check_field_error(a.lower(),b.lower(), "lower");
    check_field_error(a.upper(),b.upper(), "upper");

    for(label patchi = 0; patchi < const_cast<fvScalarMatrix&>(a).internalCoeffs().size(); ++patchi)
    {
        check_field_error(const_cast<fvScalarMatrix&>(a).internalCoeffs()[patchi],const_cast<fvScalarMatrix&>(b).internalCoeffs()[patchi], "internalCoeffs_" + std::to_string(patchi));
        check_field_error(const_cast<fvScalarMatrix&>(a).boundaryCoeffs()[patchi],const_cast<fvScalarMatrix&>(b).boundaryCoeffs()[patchi], "boundaryCoeffs_" + std::to_string(patchi));
    }
}

void check_fvmatrix_equal(const fvVectorMatrix& a,const fvVectorMatrix& b){
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
        forAll(const_cast<fvVectorMatrix&>(a).internalCoeffs().component(cmpt)(), patchi)
        {
            check_field_error(const_cast<fvVectorMatrix&>(a).internalCoeffs().component(cmpt).ref()[patchi],const_cast<fvVectorMatrix&>(b).internalCoeffs().component(cmpt).ref()[patchi], "internalCoeffs_" + std::to_string(cmpt) + "_" + std::to_string(patchi));
            check_field_error(const_cast<fvVectorMatrix&>(a).boundaryCoeffs().component(cmpt).ref()[patchi],const_cast<fvVectorMatrix&>(b).boundaryCoeffs().component(cmpt).ref()[patchi], "boundaryCoeffs_" + std::to_string(cmpt) + "_" + std::to_string(patchi));
        }
    }

}

}
