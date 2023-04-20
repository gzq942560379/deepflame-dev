#pragma once
#include <fvMesh.H>
#include <vector>
#include <string>
#include "autoPtr.H"
#include "primitiveFieldsFwd.H"
#include "FieldField.H"
#include "typeInfo.H"
#include "lduInterfaceFieldPtrsList.H"
#include "runTimeSelectionTables.H"
#include "solverPerformance.H"
#include "InfoProxy.H"

namespace Foam{

class csrMatrix{
private:
    const fvMesh& mesh_;
    //- Coefficients (not including interfaces)
    label row_;
    label col_;
    label off_diag_nnz_;
    labelList off_diag_rowptr_;
    labelList off_diag_colidx_;
    scalarField diag_value_;
    scalarField off_diag_value_;

    // map from face to lower in off_diag_value_
    // lower (nei,own)
    labelList face2lower_;
    // map from face to upper in off_diag_value_
    // lower (own,nei)
    labelList face2upper_;
    
public:
    // Static data
    // Declare name of the class and its debug switch
    ClassName("csrMatrix");

    // Constructors
    csrMatrix(const fvMesh& mesh);

    // Copy constructor
    csrMatrix(const csrMatrix& other);

    //- Destructor
    ~csrMatrix();

    //- Return the LDU mesh from which the addressing is obtained
    const fvMesh& mesh() const {return mesh_;}

    bool diagonal() const {return false;}

    bool symmetric() const {return true;}

    bool asymmetric() const {return false;}

    const scalarField& diag() const {return diag_value_;}
    scalarField& diag() {return diag_value_;}

    //- Matrix multiplication with updated interfaces.
    void Amul
    (
        scalarField&,
        const tmp<scalarField>&,
        const FieldField<Field, scalar>&,
        const lduInterfaceFieldPtrsList&,
        const direction cmpt
    ) const;
    
    //- Sum the coefficients on each row of the matrix
    void sumA
    (
        scalarField&,
        const FieldField<Field, scalar>&,
        const lduInterfaceFieldPtrsList&
    ) const;

    //- Initialise the update of interfaced interfaces
    //  for matrix operations
    void initMatrixInterfaces
    (
        const FieldField<Field, scalar>& interfaceCoeffs,
        const lduInterfaceFieldPtrsList& interfaces,
        const scalarField& psiif,
        scalarField& result,
        const direction cmpt
    ) const;

    //- Update interfaced interfaces for matrix operations
    void updateMatrixInterfaces
    (
        const FieldField<Field, scalar>& interfaceCoeffs,
        const lduInterfaceFieldPtrsList& interfaces,
        const scalarField& psiif,
        scalarField& result,
        const direction cmpt
    ) const;

    void write_pattern(const std::string& filename);


    class solver{
    protected:
        word fieldName_;
        const csrMatrix& matrix_;
        const FieldField<Field, scalar>& interfaceBouCoeffs_;
        const FieldField<Field, scalar>& interfaceIntCoeffs_;
        lduInterfaceFieldPtrsList interfaces_;

        //- Dictionary of controls
        dictionary controlDict_;
        //- Default maximum number of iterations in the solver
        static const label defaultMaxIter_;
        //- Maximum number of iterations in the solver
        label maxIter_;
        //- Minimum number of iterations in the solver
        label minIter_;
        //- Final convergence tolerance
        scalar tolerance_;
        //- Convergence tolerance relative to the initial
        scalar relTol_;
        //- Read the control parameters from the controlDict_
        virtual void readControls();
    public:
        //- Runtime type information
        virtual const word& type() const = 0;
        declareRunTimeSelectionTable
        (
            autoPtr,
            solver,
            symMatrix,
            (
                const word& fieldName,
                const csrMatrix& matrix,
                const FieldField<Field, scalar>& interfaceBouCoeffs,
                const FieldField<Field, scalar>& interfaceIntCoeffs,
                const lduInterfaceFieldPtrsList& interfaces,
                const dictionary& solverControls
            ),
            (
                fieldName,
                matrix,
                interfaceBouCoeffs,
                interfaceIntCoeffs,
                interfaces,
                solverControls
            )
        );
        declareRunTimeSelectionTable
        (
            autoPtr,
            solver,
            asymMatrix,
            (
                const word& fieldName,
                const csrMatrix& matrix,
                const FieldField<Field, scalar>& interfaceBouCoeffs,
                const FieldField<Field, scalar>& interfaceIntCoeffs,
                const lduInterfaceFieldPtrsList& interfaces,
                const dictionary& solverControls
            ),
            (
                fieldName,
                matrix,
                interfaceBouCoeffs,
                interfaceIntCoeffs,
                interfaces,
                solverControls
            )
        );

        solver(
            const word& fieldName,
            const csrMatrix& matrix,
            const FieldField<Field, scalar>& interfaceBouCoeffs,
            const FieldField<Field, scalar>& interfaceIntCoeffs,
            const lduInterfaceFieldPtrsList& interfaces,
            const dictionary& solverControls
        );

        //- Return a new solver
        static autoPtr<solver> New
        (
            const word& fieldName,
            const csrMatrix& matrix,
            const FieldField<Field, scalar>& interfaceBouCoeffs,
            const FieldField<Field, scalar>& interfaceIntCoeffs,
            const lduInterfaceFieldPtrsList& interfaces,
            const dictionary& solverControls
        );

        //- Destructor
        virtual ~solver() {}

        // Member Functions
        // Access
        const word& fieldName() const {return fieldName_;}
        const csrMatrix& matrix() const {return matrix_;}
        const FieldField<Field, scalar>& interfaceBouCoeffs() const {return interfaceBouCoeffs_;}
        const FieldField<Field, scalar>& interfaceIntCoeffs() const {return interfaceIntCoeffs_;}
        const lduInterfaceFieldPtrsList& interfaces() const{return interfaces_;}

        //- Read and reset the solver parameters from the given stream
        virtual void read(const dictionary&);
        virtual solverPerformance solve
        (
            scalarField& psi,
            const scalarField& source,
            const direction cmpt=0
        ) const = 0;

        //- Return the matrix norm used to normalise the residual for the
        //  stopping criterion
        scalar normFactor
        (
            const scalarField& psi,
            const scalarField& source,
            const scalarField& Apsi,
            scalarField& tmpField
        ) const;

    };

    //- Abstract base-class for csrMatrix smoothers
    class smoother
    {
    protected:
        // Protected data
        word fieldName_;
        const csrMatrix& matrix_;
        const FieldField<Field, scalar>& interfaceBouCoeffs_;
        const FieldField<Field, scalar>& interfaceIntCoeffs_;
        const lduInterfaceFieldPtrsList& interfaces_;

    public:
        //- Find the smoother name (directly or from a sub-dictionary)
        static word getName(const dictionary&);
        //- Runtime type information
        virtual const word& type() const = 0;
        // Declare run-time constructor selection tables
        declareRunTimeSelectionTable
        (
            autoPtr,
            smoother,
            symMatrix,
            (
                const word& fieldName,
                const csrMatrix& matrix,
                const FieldField<Field, scalar>& interfaceBouCoeffs,
                const FieldField<Field, scalar>& interfaceIntCoeffs,
                const lduInterfaceFieldPtrsList& interfaces
            ),
            (
                fieldName,
                matrix,
                interfaceBouCoeffs,
                interfaceIntCoeffs,
                interfaces
            )
        );

        declareRunTimeSelectionTable
        (
            autoPtr,
            smoother,
            asymMatrix,
            (
                const word& fieldName,
                const csrMatrix& matrix,
                const FieldField<Field, scalar>& interfaceBouCoeffs,
                const FieldField<Field, scalar>& interfaceIntCoeffs,
                const lduInterfaceFieldPtrsList& interfaces
            ),
            (
                fieldName,
                matrix,
                interfaceBouCoeffs,
                interfaceIntCoeffs,
                interfaces
            )
        );


        // Constructors
        smoother
        (
            const word& fieldName,
            const csrMatrix& matrix,
            const FieldField<Field, scalar>& interfaceBouCoeffs,
            const FieldField<Field, scalar>& interfaceIntCoeffs,
            const lduInterfaceFieldPtrsList& interfaces
        );


        // Selectors

        //- Return a new smoother
        static autoPtr<smoother> New
        (
            const word& fieldName,
            const csrMatrix& matrix,
            const FieldField<Field, scalar>& interfaceBouCoeffs,
            const FieldField<Field, scalar>& interfaceIntCoeffs,
            const lduInterfaceFieldPtrsList& interfaces,
            const dictionary& solverControls
        );


        //- Destructor
        virtual ~smoother()
        {}


        // Member Functions

        // Access

        const word& fieldName() const
        {
            return fieldName_;
        }

        const csrMatrix& matrix() const
        {
            return matrix_;
        }

        const FieldField<Field, scalar>& interfaceBouCoeffs() const
        {
            return interfaceBouCoeffs_;
        }

        const FieldField<Field, scalar>& interfaceIntCoeffs() const
        {
            return interfaceIntCoeffs_;
        }

        const lduInterfaceFieldPtrsList& interfaces() const
        {
            return interfaces_;
        }


        //- Smooth the solution for a given number of sweeps
        virtual void smooth
        (
            scalarField& psi,
            const scalarField& source,
            const direction cmpt,
            const label nSweeps
        ) const = 0;
    };


    //- Abstract base-class for csrMatrix preconditioners
    class preconditioner
    {
    protected:

        // Protected data

        //- Reference to the base-solver this preconditioner is used with
        const solver& solver_;


    public:

        //- Find the preconditioner name (directly or from a sub-dictionary)
        static word getName(const dictionary&);

        //- Runtime type information
        virtual const word& type() const = 0;


        // Declare run-time constructor selection tables

        declareRunTimeSelectionTable
        (
            autoPtr,
            preconditioner,
            symMatrix,
            (
                const solver& sol,
                const dictionary& solverControls
            ),
            (sol, solverControls)
        );

        declareRunTimeSelectionTable
        (
            autoPtr,
            preconditioner,
            asymMatrix,
            (
                const solver& sol,
                const dictionary& solverControls
            ),
            (sol, solverControls)
        );


    // Constructors

        preconditioner
        (
            const solver& sol
        )
        :
            solver_(sol)
        {}


        // Selectors

        //- Return a new preconditioner
        static autoPtr<preconditioner> New
        (
            const solver& sol,
            const dictionary& solverControls
        );


        //- Destructor
        virtual ~preconditioner()
        {}


        // Member Functions

        //- Read and reset the preconditioner parameters
        //  from the given stream
        virtual void read(const dictionary&)
        {}

        //- Return wA the preconditioned form of residual rA
        virtual void precondition
        (
            scalarField& wA,
            const scalarField& rA,
            const direction cmpt=0
        ) const = 0;

        //- Return wT the transpose-matrix preconditioned form of
        //  residual rT.
        //  This is only required for preconditioning asymmetric matrices.
        virtual void preconditionT
        (
            scalarField& wT,
            const scalarField& rT,
            const direction cmpt=0
        ) const
        {
            NotImplemented;
        }
    };

};

}

