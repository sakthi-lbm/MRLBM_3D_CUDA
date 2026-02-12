#include "save_data.cuh"

template <typename T>
void swap_endian(T &value)
{
    char *data = reinterpret_cast<char *>(&value);
    // Reverse the order of bytes in the data block
    std::reverse(data, data + sizeof(T));
}

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

void write_vtk_binary(nodeVar h_fMom, int timestep)
{
    // datafile
    std::string prefix = "data";
    std::string suffix = ".vtk";

    std::ostringstream filename_temp;
    std::string strInf3 = PATH_FILES;
    strInf3 += "/";
    strInf3 += ID_SIM;
    strInf3 += "/";
    filename_temp << strInf3 << prefix << (10000000 + timestep) << suffix;
    std::string filename = filename_temp.str();

    // Open the file in binary mode, using the stream name 'datafile'
    std::ofstream datafile(filename, std::ios::binary | std::ios::out);
    if (!datafile.is_open())
    {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }

    // --- 1. ASCII Header: Version, Title, Format ---
    datafile << "# vtk DataFile Version 3.0" << "\n";
    datafile << "2D Structured Points Data (Timestep: " << timestep << ")" << "\n";
    datafile << "BINARY" << "\n";

    // --- 2. ASCII Dataset Definition: Structured Points ---
    // Structured Points (or vtkImageData) is used for uniform grids.
    datafile << "DATASET STRUCTURED_POINTS" << "\n";

    // Define the grid dimensions (2D data, so Z dimension is 1)
    datafile << "DIMENSIONS " << NX << " " << NY << " 1" << "\n";
    // Define the origin and spacing (unit spacing for simplicity)
    datafile << "ORIGIN 0.0 0.0 0.0" << "\n";
    datafile << "SPACING 1.0 1.0 1.0" << "\n";

    // --- 3. ASCII Point Data Header (Signals start of data attribute sections) ---
    const int NUM_POINTS = NX * NY;
    datafile << "POINT_DATA " << NUM_POINTS << "\n";

    // --- 4. Write Scalar Data: Density (rho) ---
    datafile << "SCALARS rho float 1" << "\n";
    datafile << "LOOKUP_TABLE default" << "\n";

    // Write binary data for rho using the explicit nested loop structure (y then x)
    // NOTE: This is where your specific block-based indexing would be placed.
    for (int y = 0; y < NY; ++y)
    {
        for (int x = 0; x < NX; ++x)
        {
            float val = RHO_0 + h_fMom.rho[IDX_BLOCK(x % BLOCK_THREAD_X, y % BLOCK_THREAD_Y, x / BLOCK_THREAD_X, y / BLOCK_THREAD_Y)];
            float big_endian_val = val;
            swap_endian(big_endian_val); // CRITICAL: Ensure Big Endian format

            datafile.write(reinterpret_cast<char *>(&big_endian_val), sizeof(float));
        }
    }

    // --- 5. Write Scalar Data: X-Velocity (ux) ---
    datafile << "\n"
             << "SCALARS ux float 1" << "\n";
    datafile << "LOOKUP_TABLE default" << "\n";

    // Write binary data for ux using the explicit nested loop structure (y then x)
    for (int y = 0; y < NY; ++y)
    {
        for (int x = 0; x < NX; ++x)
        {
            float val = h_fMom.ux[IDX_BLOCK(x % BLOCK_THREAD_X, y % BLOCK_THREAD_Y, x / BLOCK_THREAD_X, y / BLOCK_THREAD_Y)];
            float big_endian_val = val;
            swap_endian(big_endian_val);
            datafile.write(reinterpret_cast<char *>(&big_endian_val), sizeof(float));
        }
    }

    // --- 6. Write Scalar Data: Y-Velocity (uy) ---
    datafile << "\n"
             << "SCALARS uy float 1" << "\n";
    datafile << "LOOKUP_TABLE default" << "\n";

    // Write binary data for uy using the explicit nested loop structure (y then x)
    for (int y = 0; y < NY; ++y)
    {
        for (int x = 0; x < NX; ++x)
        {
            float val = h_fMom.uy[IDX_BLOCK(x % BLOCK_THREAD_X, y % BLOCK_THREAD_Y, x / BLOCK_THREAD_X, y / BLOCK_THREAD_Y)];
            float big_endian_val = val;
            swap_endian(big_endian_val);
            datafile.write(reinterpret_cast<char *>(&big_endian_val), sizeof(float));
        }
    }

    // --- 7. Finalization ---
    datafile.close();
    std::cout << "Successfully wrote " << filename << " (" << NUM_POINTS << " points)" << std::endl;
}

void write_vti(nodeVar data, int timestep)
{
    std::ostringstream filename_temp;
    filename_temp << PATH_FILES << "/" << ID_SIM << "/data"
                  << (10000000 + timestep) << ".vti";
    std::string filename = filename_temp.str();

    const uint64_t total_points = NX * NY;
    const uint64_t bytes_per_field = total_points * sizeof(float);

    const char *vtkType = (sizeof(float) == 4 ? "Float32" : "Float64");

    std::ofstream f(filename, std::ios::binary);
    if (!f)
    {
        std::cerr << "Cannot open " << filename << "\n";
        return;
    }

    // ------------------------------------------------------------
    // XML HEADER
    // ------------------------------------------------------------
    uint64_t offset = 0;

    f << "<?xml version=\"1.0\"?>\n"
         "<VTKFile type=\"ImageData\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\">\n"
         "  <ImageData WholeExtent=\"0 "
      << NX - 1
      << " 0 " << NY - 1
      << " 0 0\" Origin=\"0 0 0\" Spacing=\"1 1 1\">\n"
         "    <Piece Extent=\"0 "
      << NX - 1
      << " 0 " << NY - 1
      << " 0 0\">\n"
         "      <PointData>\n"

         "        <DataArray type=\""
      << vtkType
      << "\" Name=\"rho\" format=\"appended\" offset=\"" << offset << "\"/>\n";
    offset += 8 + bytes_per_field;

    f << "        <DataArray type=\"" << vtkType
      << "\" Name=\"ux\" format=\"appended\" offset=\"" << offset << "\"/>\n";
    offset += 8 + bytes_per_field;

    f << "        <DataArray type=\"" << vtkType
      << "\" Name=\"uy\" format=\"appended\" offset=\"" << offset << "\"/>\n"
                                                                     "      </PointData>\n"
                                                                     "    </Piece>\n"
                                                                     "  </ImageData>\n"

                                                                     "  <AppendedData encoding=\"raw\">\n_";

    f.flush(); // <-- CRITICAL

    // ------------------------------------------------------------
    // BINARY SECTION
    // ------------------------------------------------------------

    // Helper lambda
    auto write_field = [&](auto &A)
    {
        uint64_t header = bytes_per_field;
        f.write((char *)&header, 8);

        for (int y = 0; y < NY; y++)
            for (int x = 0; x < NX; x++)
            {
                float v = A[IDX_BLOCK(
                    x % BLOCK_THREAD_X,
                    y % BLOCK_THREAD_Y,
                    x / BLOCK_THREAD_X,
                    y / BLOCK_THREAD_Y)];

                f.write((char *)&v, sizeof(float));
            }
    };

    // Density
    {
        uint64_t header = bytes_per_field;
        f.write((char *)&header, 8);
        for (int y = 0; y < NY; y++)
            for (int x = 0; x < NX; x++)
            {
                float v = RHO_0 + data.rho[IDX_BLOCK(x % BLOCK_THREAD_X, y % BLOCK_THREAD_Y, x / BLOCK_THREAD_X, y / BLOCK_THREAD_Y)];
                f.write((char *)&v, sizeof(float));
            }
    }

    // Velocity X
    write_field(data.ux);

    // Velocity Y
    write_field(data.uy);

    // ------------------------------------------------------------
    // XML FOOTER
    // ------------------------------------------------------------
    f << "\n  </AppendedData>\n</VTKFile>\n";

    f.close();
}

void write_vti_cylinder(nodeVar data, int timestep)
{
    std::ostringstream filename_temp;
    filename_temp << PATH_FILES << "/" << ID_SIM << "/data" << (10000000 + timestep) << ".vti";
    std::string filename = filename_temp.str();

    const uint64_t total_points = NX * NY;
    const uint64_t bytes_per_field = total_points * sizeof(float);
    const char *vtkType = "Float32";

    // Define the fields we want to write
    std::vector<std::string> field_names = {"rho", "ux", "uy"};

    std::ofstream f(filename, std::ios::binary);
    if (!f)
        return;

    // --- XML HEADER ---
    f << "<?xml version=\"1.0\"?>\n"
      << "<VTKFile type=\"ImageData\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\">\n"
      << "  <ImageData WholeExtent=\"0 " << NX - 1 << " 0 " << NY - 1 << " 0 0\" Origin=\"0 0 0\" Spacing=\"1 1 1\">\n"
      << "    <Piece Extent=\"0 " << NX - 1 << " 0 " << NY - 1 << " 0 0\">\n"
      << "      <PointData>\n";

    // Automatically calculate offsets
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

    // --- HELPER FUNCTION FOR DATA ---
    auto write_data = [&](auto compute_func)
    {
        uint64_t header = bytes_per_field;
        f.write((char *)&header, 8);
        for (size_t y = 0; y < NY; y++)
        {
            for (size_t x = 0; x < NX; x++)
            {
                size_t idx = IDX_BLOCK(x % BLOCK_THREAD_X, y % BLOCK_THREAD_Y, x / BLOCK_THREAD_X, y / BLOCK_THREAD_Y);
                float val = compute_func(x, y, idx);
                f.write((char *)&val, sizeof(float));
            }
        }
    };

    // --- BINARY SECTION ---

    // 1. rho (x and y unused)
    write_data([&](int, int, size_t idx)
               { return RHO_0 + data.rho[idx]; });

    // 2. ux (x and y unused)
    write_data([&](int, int, size_t idx)
               { return (data.nodeType[idx] == SOLID) ? NAN : (float)(data.ux[idx]); });

    // 3. uy (x and y unused)
    write_data([&](int, int, size_t idx)
               { return (data.nodeType[idx] == SOLID) ? NAN : (float)(data.uy[idx]); });


    // --- FOOTER ---
    f << "\n  </AppendedData>\n</VTKFile>\n";
    f.close();
}

