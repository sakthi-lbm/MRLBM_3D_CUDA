#include "saveData.cuh"

void write_master_pvd()
{
    // Creating master.p3d file
    std::string strInf = PATH_FILES;
    strInf += "/";
    strInf += ID_SIM;
    strInf += "/";
    strInf += "master.pvd"; // generate file name (with path)
    std::string prefix = "data";
    std::string suffix = ".vti";
    std::ostringstream master_file;
    master_file << strInf;
    std::ofstream out(master_file.str());
    out << "<VTKFile type=\"Collection\" version=\"0.1\">" << std::endl;
    out << "\t<Collection>" << std::endl;
    for (int iter = 0; iter <= MAX_ITER; iter++)
    {
        if (iter % MACR_SAVE == 0)
        {
            std::ostringstream filename_temp;
            filename_temp << prefix << (10000000 + iter) << suffix;
            std::string filename = filename_temp.str();

            if (out.is_open())
            {
                out << "\t\t<DataSet timestep=\"" << iter / MACR_SAVE << "\" file=\"" << filename << "\" />" << std::endl;
                ;
            }
            else
            {
                std::cerr << "Failed to open file: master.p3d" << std::endl;
            }
        }
    }
    out << "\t</Collection>" << std::endl;
    out << "</VTKFile>" << std::endl;
    out.close();
}

void write_vti_3d(nodeVar data, int timestep)
{
    std::ostringstream filename_temp;
    filename_temp << PATH_FILES << "/" << ID_SIM << "/data" << (10000000 + timestep) << ".vti";
    std::string filename = filename_temp.str();

    // --- 1. Update total points for 3D ---
    const uint64_t total_points = (uint64_t)NX * NY * NZ;
    const uint64_t bytes_per_field = total_points * sizeof(float);
    const char *vtkType = "Float32";

    std::vector<std::string> field_names = {"rho", "ux", "uy", "uz"}; // Added uz for 3D

    std::ofstream f(filename, std::ios::binary);
    if (!f)
        return;

    // --- 2. Update XML Header Extents (0 to N-1) ---
    f << "<?xml version=\"1.0\"?>\n"
      << "<VTKFile type=\"ImageData\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\">\n"
      << "  <ImageData WholeExtent=\"0 " << NX - 1 << " 0 " << NY - 1 << " 0 " << NZ - 1 << "\" Origin=\"0 0 0\" Spacing=\"1 1 1\">\n"
      << "    <Piece Extent=\"0 " << NX - 1 << " 0 " << NY - 1 << " 0 " << NZ - 1 << "\">\n"
      << "      <PointData>\n";

    uint64_t offset = 0;
    for (const auto &name : field_names)
    {
        f << "        <DataArray type=\"" << vtkType << "\" Name=\"" << name << "\" format=\"appended\" offset=\"" << offset << "\"/>\n";
        offset += 8 + bytes_per_field;
    }

    f << "      </PointData>\n"
      << "    </Piece>\n"
      << "  </ImageData>\n"
      << "  <AppendedData encoding=\"raw\">\n_";
    f.flush();

    // --- 3. Helper Function with 3-Level Loop ---
    auto write_data = [&](auto compute_func)
    {
        uint64_t header = bytes_per_field;
        f.write((char *)&header, 8);
        for (size_t z = 0; z < NZ; z++) // Outer loop: Z
        {
            for (size_t y = 0; y < NY; y++) // Middle loop: Y
            {
                for (size_t x = 0; x < NX; x++) // Inner loop: X
                {
                    // Ensure your IDX_BLOCK or indexing function matches your 3D layout
                    size_t idx = IDX_BLOCK(x % BLOCK_THREAD_X,
                                           y % BLOCK_THREAD_Y,
                                           z % BLOCK_THREAD_Z,
                                           x / BLOCK_THREAD_X,
                                           y / BLOCK_THREAD_Y,
                                           z / BLOCK_THREAD_Z);
                    float val = compute_func(idx);
                    f.write((char *)&val, sizeof(float));
                }
            }
        }
    };

    // --- 4. Binary Section ---
    write_data([&](size_t idx)
               { return RHO_0 + data.rho[idx]; });
    write_data([&](size_t idx)
               { return (getType(data.nodeType[idx]) == NODE_SOLID) ? NAN : (float)data.ux[idx]; });
    write_data([&](size_t idx)
               { return (getType(data.nodeType[idx]) == NODE_SOLID) ? NAN : (float)data.uy[idx]; });
    write_data([&](size_t idx)
               { return (getType(data.nodeType[idx]) == NODE_SOLID) ? NAN : (float)data.uz[idx]; });

    f << "\n  </AppendedData>\n</VTKFile>\n";
    f.close();
}

void write_vti_3d_annulus(nodeVar data, int timestep)
{
    std::ostringstream filename_temp;
    filename_temp << PATH_FILES << "/" << ID_SIM << "/data" << (10000000 + timestep) << ".vti";
    std::string filename = filename_temp.str();

    // --- 1. Update total points for 3D ---
    const uint64_t total_points = (uint64_t)NX * NY * NZ;
    const uint64_t bytes_per_field = total_points * sizeof(float);
    const char *vtkType = "Float32";

    std::vector<std::string> field_names = {"rho", "ux", "uy", "uz", "u_theta", "u_radial"}; // Added uz for 3D

    std::ofstream f(filename, std::ios::binary);
    if (!f)
        return;

    // --- 2. Update XML Header Extents (0 to N-1) ---
    f << "<?xml version=\"1.0\"?>\n"
      << "<VTKFile type=\"ImageData\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\">\n"
      << "  <ImageData WholeExtent=\"0 " << NX - 1 << " 0 " << NY - 1 << " 0 " << NZ - 1 << "\" Origin=\"0 0 0\" Spacing=\"1 1 1\">\n"
      << "    <Piece Extent=\"0 " << NX - 1 << " 0 " << NY - 1 << " 0 " << NZ - 1 << "\">\n"
      << "      <PointData>\n";

    uint64_t offset = 0;
    for (const auto &name : field_names)
    {
        f << "        <DataArray type=\"" << vtkType << "\" Name=\"" << name << "\" format=\"appended\" offset=\"" << offset << "\"/>\n";
        offset += 8 + bytes_per_field;
    }

    f << "      </PointData>\n"
      << "    </Piece>\n"
      << "  </ImageData>\n"
      << "  <AppendedData encoding=\"raw\">\n_";
    f.flush();

    // --- 3. Helper Function with 3-Level Loop ---
    auto write_data = [&](auto compute_func)
    {
        uint64_t header = bytes_per_field;
        f.write((char *)&header, 8);

        for (size_t z = 0; z < NZ; z++)
        {
            for (size_t y = 0; y < NY; y++)
            {
                for (size_t x = 0; x < NX; x++)
                {
                    size_t idx = IDX_BLOCK(
                        x % BLOCK_THREAD_X,
                        y % BLOCK_THREAD_Y,
                        z % BLOCK_THREAD_Z,
                        x / BLOCK_THREAD_X,
                        y / BLOCK_THREAD_Y,
                        z / BLOCK_THREAD_Z);

                    float val = compute_func(x, y, z, idx);
                    f.write((char *)&val, sizeof(float));
                }
            }
        }
    };

    auto get_cylindrical = [&](size_t idx, int x, int y, bool get_theta)
    {
        nodeType_t node = data.nodeType[idx];
        if (getType(node) == NODE_SOLID)
            return (float)NAN;

        float ux = data.ux[idx];
        float uy = data.uy[idx];

        float dx = (float)x - XC;
        float dy = (float)y - YC;

        float r2 = dx * dx + dy * dy;
        if (r2 < 1e-12f)
            return 0.0f;

        float inv_r = 1.0f / sqrtf(r2);
        float cos_t = dx * inv_r;
        float sin_t = dy * inv_r;

        // u_theta = -ux*sinθ + uy*cosθ
        if (get_theta)
            return (-ux * sin_t + uy * cos_t) / (float)VEL_NORM;

        // u_radial = ux*cosθ + uy*sinθ
        return (ux * cos_t + uy * sin_t) / (float)VEL_NORM;
    };

    // --- 4. Binary Section ---
    write_data([&](int, int, int, size_t idx)
               { return RHO_0 + data.rho[idx]; });

    write_data([&](int, int, int, size_t idx)
               { return (getType(data.nodeType[idx]) == NODE_SOLID) ? NAN : (float)data.ux[idx]; });

    write_data([&](int, int, int, size_t idx)
               { return (getType(data.nodeType[idx]) == NODE_SOLID) ? NAN : (float)data.uy[idx]; });

    write_data([&](int, int, int, size_t idx)
               { return (getType(data.nodeType[idx]) == NODE_SOLID) ? NAN : (float)data.uz[idx]; });

    write_data([&](int x, int y, int, size_t idx)
               { return get_cylindrical(idx, x, y, true); });

    write_data([&](int x, int y, int, size_t idx)
               { return get_cylindrical(idx, x, y, false); });

    f << "\n  </AppendedData>\n</VTKFile>\n";
    f.close();
}