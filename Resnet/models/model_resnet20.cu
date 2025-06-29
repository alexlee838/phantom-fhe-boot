#include "model_resnet20.cuh"

void PrePareResNet20(DNN& model, PhantomSecretKey& secret_key) {
    std::vector<int32_t> vec;
    model.AddRotationIndicesTo(vec, 32, 3, 0);
    model.AddRotationIndicesTo(vec, 16, 3, 1);
    model.AddRotationIndicesTo(vec, 8, 3, 2);
    model.AddAvgPoolRotationsTo(vec, 8, 2);
    model.BuildGaloisKey(secret_key, vec);
    model.RelinKeyGen(secret_key);

    return;

}

void debug_print(const TensorCT& x, DNN* model = nullptr, PhantomSecretKey* secret_key = nullptr, PhantomContext* context = nullptr, PhantomCKKSEncoder* encoder = nullptr, bool set = false) {
    static PhantomSecretKey* sk = nullptr;
    static PhantomContext* ctx = nullptr;
    static PhantomCKKSEncoder* ecd = nullptr;
    static DNN* md = nullptr;


    if (set) {
        sk = secret_key;
        ctx = context;
        ecd = encoder;
        md = model;
    }

    else {
        if (!sk || !ctx || !ecd || !md) {
            std::cerr << "❌ Error: debug_print called before setup.\n";
        }
        auto vec = md->DecTensor(x, *sk);

        auto [min_val, max_val] = find_min_max(vec);

        std::cout << "Min: " << min_val << ", Max: " << max_val << std::endl;
    }
}



TensorCT ResNet20_infer(const TensorCT& input, DNN& model, FHECKKSRNS& bootstrapper, const PhantomContext& context, const std::string& weight_dir) {

    TensorCT x = input;

    // conv1 + bn1
    auto conv1_w = LoadWeight4D(weight_dir + "/conv1_weight.npy");
    auto bn1 = LoadBN(weight_dir, "bn1");

    x = model.Conv(x, conv1_w, 1);
    x = model.BatchNorm(x, bn1.weight, bn1.bias, bn1.mean, bn1.var);
    x = model.ReluComposite(x, bootstrapper); //Min: -0.196985, Max: 0.881967


    // ---- Layer 1 (3 blocks, no downsample) ----
    for (int i = 0; i < 3; ++i) {
        std::cout << "Processing Layer 1 Block " << i + 1 << std::endl;

        TensorCT x_res = x;
        std::string prefix = "/layer1_" + std::to_string(i);

        auto w1 = LoadWeight4D(weight_dir + prefix + "_conv1_weight.npy");
        auto bw1 = LoadBN(weight_dir, prefix + "_bn1");
        
        x = model.Conv(x, w1, 1);
        x = model.BatchNorm(x, bw1.weight, bw1.bias, bw1.mean, bw1.var);
        x = model.ReluComposite(x, bootstrapper);

        auto w2 = LoadWeight4D(weight_dir + prefix + "_conv2_weight.npy");
        auto bw2 = LoadBN(weight_dir, prefix + "_bn2");

        x = model.Conv(x, w2, 1);
        x = model.BatchNorm(x, bw2.weight, bw2.bias, bw2.mean, bw2.var);
        x = model.Add(x, x_res);  // Encrypted residual addition
        x = model.ReluComposite(x, bootstrapper);
    } 

    // ---- Layer 2 (3 blocks, downsample at block 0) ----
    for (int i = 0; i < 3; ++i) {
        std::cout << "Processing Layer 2 Block " << i + 1 << std::endl;

        std::string prefix = "/layer2_" + std::to_string(i);
        TensorCT x_skip = x;
        auto w1 = LoadWeight4D(weight_dir + prefix + "_conv1_weight.npy");
        auto bw1 = LoadBN(weight_dir, prefix + "_bn1");

        TensorCT x_main = model.Conv(x, w1, (i == 0) ? 2 : 1); // i==0, Min: -1.36329, Max: 1.30132

        x_main = model.BatchNorm(x_main, bw1.weight, bw1.bias, bw1.mean, bw1.var);

        x_main = model.ReluComposite(x_main, bootstrapper);

        auto w2 = LoadWeight4D(weight_dir + prefix + "_conv2_weight.npy");
        auto bw2 = LoadBN(weight_dir, prefix + "_bn2");
        x_main = model.Conv(x_main, w2, 1);
        x_main = model.BatchNorm(x_main, bw2.weight, bw2.bias, bw2.mean, bw2.var);

        if (i == 0) {
            auto w_skip = LoadWeight4D(weight_dir + prefix + "_downsample_0_weight.npy");
            auto bn_skip = LoadBN(weight_dir, prefix + "_downsample_1");
            x_skip = model.Conv(x_skip, w_skip, 2);
            x_skip = model.BatchNorm(x_skip, bn_skip.weight, bn_skip.bias, bn_skip.mean, bn_skip.var);
        }

        x = model.Add(x_main, x_skip);

        x = model.ReluComposite(x, bootstrapper);

    } // Min: -0.184489, Max: 1.63471

    // ---- Layer 3 (same structure as Layer 2) ----
    for (int i = 0; i < 3; ++i) {
        std::cout << "Processing Layer 3 Block " << i + 1 << std::endl;

        std::string prefix = "/layer3_" + std::to_string(i);
        TensorCT x_skip = x;
        auto w1 = LoadWeight4D(weight_dir + prefix + "_conv1_weight.npy");
        auto bw1 = LoadBN(weight_dir, prefix + "_bn1");

        TensorCT x_main = model.Conv(x, w1, (i == 0) ? 2 : 1);

        x_main = model.BatchNorm(x_main, bw1.weight, bw1.bias, bw1.mean, bw1.var);

        x_main = model.ReluComposite(x_main, bootstrapper); 


        auto w2 = LoadWeight4D(weight_dir + prefix + "_conv2_weight.npy");
        auto bw2 = LoadBN(weight_dir, prefix + "_bn2");
        x_main = model.Conv(x_main, w2, 1);
        x_main = model.BatchNorm(x_main, bw2.weight, bw2.bias, bw2.mean, bw2.var);

        if (i == 0) {
            auto w_skip = LoadWeight4D(weight_dir + prefix + "_downsample_0_weight.npy");
            auto bn_skip = LoadBN(weight_dir, prefix + "_downsample_1");
            x_skip = model.Conv(x_skip, w_skip, 2);
            x_skip = model.BatchNorm(x_skip, bn_skip.weight, bn_skip.bias, bn_skip.mean, bn_skip.var);
        }

        x = model.Add(x_main, x_skip);

        x = model.ReluComposite(x, bootstrapper);
    }

    // ---- Final FC ----
    std::cout << "Processing Final Fully Connected Layer" << std::endl;
    auto fc_w = LoadWeight2D(weight_dir + "/fc_weight.npy");  // shape (10, 64)
    auto fc_b = LoadWeight1D(weight_dir + "/fc_bias.npy");
    x = model.AvgPoolFullCon(x, fc_w, fc_b);

    return x;
}
