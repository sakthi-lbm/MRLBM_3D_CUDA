#include "restart_io.cuh"

void write_checkpoint(nodeVar &hMom, haloData &fHalo, int iter)
{
    std::ofstream file(construct_path(PATH_FILES, ID_SIM, "checkpoint/restart.bin"), std::ios::binary);

    file.write((char *)&iter, sizeof(int));

    write_moments(file, hMom);
    write_halo(file, fHalo);

    file.close();
    std::cout << "Saving checkpoint at iteration: " << iter << std::endl;
}

int read_checkpoint(nodeVar &hMom, haloData &fHalo)
{
    std::ifstream file(construct_path(PATH_FILES, ID_SIM, "checkpoint/restart.bin"), std::ios::binary);

    if (!file)
        return 0;

    int iter;

    file.read((char *)&iter, sizeof(int));

    file.read((char *)hMom.rho, sizeof(real) * NUM_LBM_NODES);
    file.read((char *)hMom.ux, sizeof(real) * NUM_LBM_NODES);
    file.read((char *)hMom.uy, sizeof(real) * NUM_LBM_NODES);
    file.read((char *)hMom.uz, sizeof(real) * NUM_LBM_NODES);

    file.read((char *)hMom.mxx, sizeof(real) * NUM_LBM_NODES);
    file.read((char *)hMom.myy, sizeof(real) * NUM_LBM_NODES);
    file.read((char *)hMom.mzz, sizeof(real) * NUM_LBM_NODES);

    file.read((char *)hMom.mxy, sizeof(real) * NUM_LBM_NODES);
    file.read((char *)hMom.mxz, sizeof(real) * NUM_LBM_NODES);
    file.read((char *)hMom.myz, sizeof(real) * NUM_LBM_NODES);

    file.read((char *)fHalo.X_WEST, sizeof(real) * NUM_HALO_FACE_YZ * QF);
    file.read((char *)fHalo.X_EAST, sizeof(real) * NUM_HALO_FACE_YZ * QF);

    file.read((char *)fHalo.Y_SOUTH, sizeof(real) * NUM_HALO_FACE_XZ * QF);
    file.read((char *)fHalo.Y_NORTH, sizeof(real) * NUM_HALO_FACE_XZ * QF);

    file.read((char *)fHalo.Z_BACK, sizeof(real) * NUM_HALO_FACE_XY * QF);
    file.read((char *)fHalo.Z_FRONT, sizeof(real) * NUM_HALO_FACE_XY * QF);

    file.close();

    return iter;
}

void write_moments(std::ofstream &file, nodeVar &hMom)
{
    file.write((char *)hMom.rho, sizeof(real) * NUM_LBM_NODES);
    file.write((char *)hMom.ux, sizeof(real) * NUM_LBM_NODES);
    file.write((char *)hMom.uy, sizeof(real) * NUM_LBM_NODES);
    file.write((char *)hMom.uz, sizeof(real) * NUM_LBM_NODES);

    file.write((char *)hMom.mxx, sizeof(real) * NUM_LBM_NODES);
    file.write((char *)hMom.myy, sizeof(real) * NUM_LBM_NODES);
    file.write((char *)hMom.mzz, sizeof(real) * NUM_LBM_NODES);

    file.write((char *)hMom.mxy, sizeof(real) * NUM_LBM_NODES);
    file.write((char *)hMom.mxz, sizeof(real) * NUM_LBM_NODES);
    file.write((char *)hMom.myz, sizeof(real) * NUM_LBM_NODES);
}

void write_halo(std::ofstream &file, haloData &fHalo)
{
    file.write((char *)fHalo.X_WEST, sizeof(real) * NUM_HALO_FACE_YZ * QF);
    file.write((char *)fHalo.X_EAST, sizeof(real) * NUM_HALO_FACE_YZ * QF);

    file.write((char *)fHalo.Y_SOUTH, sizeof(real) * NUM_HALO_FACE_XZ * QF);
    file.write((char *)fHalo.Y_NORTH, sizeof(real) * NUM_HALO_FACE_XZ * QF);

    file.write((char *)fHalo.Z_BACK, sizeof(real) * NUM_HALO_FACE_XY * QF);
    file.write((char *)fHalo.Z_FRONT, sizeof(real) * NUM_HALO_FACE_XY * QF);
}
