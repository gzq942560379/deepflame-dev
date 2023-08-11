


const auto& cellCentres = mesh.cellCentres();

std::vector<std::tuple<label,label,label>> points;

forAll(cellCentres, i){
    points.push_back(std::make_tuple(static_cast<label>(cellCentres[i][0] * 1e6), static_cast<label>(cellCentres[i][1] * 1e6), static_cast<label>(cellCentres[i][2] * 1e6)));
}

label z_size = 1;
for(size_t i = 1; i < points.size(); ++i){
    if(std::get<2>(points[i]) == std::get<2>(points[i-1])){
        z_size += 1;
    }else{
        break;
    }
}

label z_count = mesh.nCells() / z_size;

const labelUList& face_ptr = mesh.lduAddr().ownerStartAddr();
const labelUList& l = mesh.lduAddr().lowerAddr();
const labelUList& u = mesh.lduAddr().upperAddr();

labelList cell_scheduling(z_count + 1);
labelList face_scheduling(z_count + 1);

cell_scheduling[0] = 0;
face_scheduling[0] = 0;

for(label i = 0 ; i < z_count; i++){
    cell_scheduling[i+1] = cell_scheduling[i] + z_size;
    face_scheduling[i+1] = face_ptr[cell_scheduling[i+1]];
}

Info << "z_size : " << z_size << endl;
Info << "z_count : " << z_count << endl;

// forAll(cell_scheduling, i){
//     Info << "cell_scheduling[i] : " << cell_scheduling[i] << endl;
//     Info << "face_scheduling[i] : " << face_scheduling[i] << endl;
// }

