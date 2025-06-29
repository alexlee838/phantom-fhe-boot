#include "weight_loader.cuh"

Weight4D LoadWeight4D(const std::string& path) {
    cnpy::NpyArray arr = cnpy::npy_load(path);

    if (arr.word_size != sizeof(float)) {
        throw std::runtime_error("Expected float32 .npy file");
    }

    if (arr.shape.size() != 4) {
        throw std::runtime_error("Expected 4D weight tensor");
    }

    // PyTorch layout: [out_ch, in_ch, kH, kW]
    int out_ch = arr.shape[0];
    int in_ch = arr.shape[1];
    int kH = arr.shape[2];
    int kW = arr.shape[3];

    const float* data = arr.data<float>();

    // Target layout: [kH][kW][in_ch][out_ch]
    Weight4D weight(kH, std::vector<std::vector<std::vector<double>>>(
        kW, std::vector<std::vector<double>>(
            in_ch, std::vector<double>(out_ch)
        )
    ));

    size_t idx = 0;
    for (int o = 0; o < out_ch; ++o)
        for (int i = 0; i < in_ch; ++i)
            for (int h = 0; h < kH; ++h)
                for (int w = 0; w < kW; ++w) {
                    double val = static_cast<double>(data[idx++]);
                    weight[h][w][i][o] = val;
                }

    return weight;
}

std::vector<std::vector<double>> LoadWeight2D(const std::string& path) {
    cnpy::NpyArray arr = cnpy::npy_load(path);
    auto* data = arr.data<float>();
    auto& shape = arr.shape;

    if (shape.size() != 2) {
        throw std::runtime_error("Expected 2D .npy array: " + path);
    }

    int rows = shape[0], cols = shape[1];
    std::vector<std::vector<double>> result(rows, std::vector<double>(cols));

    size_t idx = 0;
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            result[i][j] = static_cast<double>(data[idx++]);

    return result;
}


std::vector<double> LoadWeight1D(const std::string& path) {
    cnpy::NpyArray arr = cnpy::npy_load(path);
    auto* data = arr.data<float>();
    auto& shape = arr.shape;

    if (shape.size() != 1) {
        throw std::runtime_error("Expected 1D .npy array: " + path);
    }

    std::vector<double> result(shape[0]);
    for (size_t i = 0; i < shape[0]; ++i) {
        result[i] = static_cast<double>(data[i]);
    }

    return result;
}

BatchNormParams LoadBN(const std::string& dir, const std::string& prefix) {
    BatchNormParams bn;
    bn.weight = LoadWeight1D(dir + "/" + prefix + "_weight.npy");
    bn.bias = LoadWeight1D(dir + "/" + prefix + "_bias.npy");
    bn.mean = LoadWeight1D(dir + "/" + prefix + "_running_mean.npy");
    bn.var = LoadWeight1D(dir + "/" + prefix + "_running_var.npy");
    return bn;
}

std::vector<std::vector<std::vector<double>>> load_next_cifar_image(const std::string& npy_path) {
    static size_t image_index = 0; // static counter

    const int num_images = 10000;
    const int c_i = 3;
    const int w = 32;

    if (image_index >= num_images) {
        std::cerr << "All images loaded.\n";
        exit(1);
    }

    // Load the entire numpy file
    cnpy::NpyArray arr = cnpy::npy_load(npy_path);
    float* data = arr.data<float>();  // float32 in file

    // Validate shape
    if (arr.shape[0] != num_images || arr.shape[1] != c_i || arr.shape[2] != w || arr.shape[3] != w) {
        throw std::runtime_error("Unexpected shape in CIFAR-10 image array");
    }

    // Allocate result image: [w][w][c_i]
    std::vector<std::vector<std::vector<double>>> image(w, std::vector<std::vector<double>>(w, std::vector<double>(c_i, 0.0)));

    // Copy data from flattened [n, c, h, w] format
    size_t base_offset = image_index * c_i * w * w;

    for (int c = 0; c < c_i; ++c) {
        for (int y = 0; y < w; ++y) {
            for (int x = 0; x < w; ++x) {
                size_t offset = base_offset + c * w * w + y * w + x;
                image[y][x][c] = static_cast<double>(data[offset]);
            }
        }
    }

    image_index++;
    return image;
}