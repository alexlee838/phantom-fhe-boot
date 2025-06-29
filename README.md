# PhantomFHE (Modified Fork)

This is a modified fork of [PhantomFHE](https://github.com/encryptorion-lab/phantom-fhe), an open-source library for Fully Homomorphic Encryption (FHE) with native GPU acceleration. This fork includes custom changes to support additional functionality and improvements.

📘 **Project Note**: This repository was developed as part of an undergraduate graduation project (2025) in the Department of Electrical and Computer Engineering at Seoul National University.

---

## 🔧 Modifications

This fork introduces the following enhancements over the original PhantomFHE:

- **CKKS Bootstrapping Support**  
  Integrated support for bootstrapping in the CKKS scheme.  
  *Implementation adapted from OpenFHE.*

- **Privacy-Preserving Machine Learning (PPML) Kernels**  
  Added neural network inference kernels, including Convolution, Softmax, Relu, Average Pooling, Fully Connected Layer, designed for encrypted inference.  
  *Based on concepts from:  
  "Privacy-Preserving Machine Learning With Fully Homomorphic Encryption for Deep Neural Network"  
  by Joon-Woo Lee, Hyungchul Kang, Jieun Eom, Maxim Deryabin, Yongwoo Lee, Woosuk Choi, Eunsang Lee, Junghyun Lee (Graduate Student Member, IEEE),  
  Donghoon Yoo, Young-Sik Kim (Member, IEEE), and Jong-Seon No (Fellow, IEEE).  
  Published in: IEEE Access, Volume 10, 2022.  
  Available at: [https://ieeexplore.ieee.org/document/9734024](https://ieeexplore.ieee.org/document/9734024)*


---

## 📄 Documentation

Refer to the [original PhantomFHE documentation](https://encryptorion-lab.gitbook.io/phantom-fhe/) for installation, usage, and background information. This fork largely follows the same interface.

---

## 📦 Build & Installation

```bash
# Clone this repository
git clone https://github.com/alexlee838/phantom-fhe-boot.git
cd phantom-boot

# Run individual examples (each script builds the library before execution)
./run_test.sh    # Builds and runs example.cu
./run_boot.sh    # Builds and runs bootstrapping_example.cu
./run_dnn.sh     # Builds and runs dnn_example.cu
```
---

## License

PhantomFHE is released under GPLv3 license. See [LICENSE](LICENSE) for more information.

Some files contain the modified code from [Microsoft SEAL](https://github.com/microsoft/SEAL). These codes are released
under MIT License. See [MIT License](https://github.com/microsoft/SEAL/blob/main/LICENSE) for more information.

Some files contain the modified code from [OpenFHE](https://github.com/openfheorg/openfhe-development). These codes are
released under BSD 2-Clause License.
See [BSD 2-Clause License](https://github.com/openfheorg/openfhe-development/blob/main/LICENSE) for more information.

