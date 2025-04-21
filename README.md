# OpenCIMTC

This project is a software toolchain for memristor-based compute-in-memory (CIM) systems, encompassing compilers, optimizers, simulators, and training modules. It assists developers in the entire process from ONNX model to chip deployment. The distinctive features of this project include:

1. Providing a high-level intermediate representation (CIM-IR) tailored for CIM systems;
2. Automatically generating hardware-aware training codes;
3. Automatically optimizing hardware parameters for inferencing;
4. Automatically implementing weight placement;
5. Automatically generating hardware inference functions.

For further details, refer to the paper [*A full-stack memristor-based computation-in-memory system with software-hardware co-development.*](https://www.nature.com/articles/s41467-025-57183-0)

# Install

```
git clone git@github.com:Tsinghua-LEMON-Lab/OpenCIMTC.git
cd OpenCIMTC
pip install -e .
```

Additionally, this project depends on other third-party Python packages. Please check requirement.txt for the necessary packages or install them using the following command:

```
pip install -r requirement.txt
```

# System Requirements

We have tested this on `Windows 10` and  `Ubuntu 20.04.1 LTS`

# Examples

Converting an ONNX model into executable code for a CIM system primarily involves three main processes: retraining with awareness of hardware computation processes(post-deployment training), optimization of hardware tunable parameters, and generation of inference codes. Since direct access to real hardware platforms may not be feasible, this example focuses on generating code that can be run on a simulator.
Here, we use ResNet-32 as an example to illustrate the full process. We have prepared a full-precision trained ONNX model `resnet32.onnx` in the `test/resnet32/` directory.

## 1. Post-deployment Training

Firstly, use the compiler to convert `resnet32.onnx` into CIM-IR and extract the weight values from the ONNX file as the initial weights for retraining. Go to the `test` directory and run the command:

```
python cimcc.py -o resnet32/resnet32.onnx --to_ir
```

Upon successful execution, an `ir` directory and a weight file named `resnet32_float_weight.pth.tar` will be generated in the `resnet32` directory. Additionally, a file named `resnet32_mapped_ir.yaml` will be created in the `ir` directory. Using this IR file, continue to generate trainable code that includes hardware execution processes by running the command:

```
python cimcc.py -i resnet32/ir/resnet32_mapped_ir.yaml --to_train
```

After that, a `scripts` directory will be created in the `resnet32` directory, with a file named `resnet32_mapped_ir_PDT.py` inside it. We have also provided example configurations for training in the `resnet32/training_config` directory, including `quantization_a4w4_wo_noise.yaml` for quantization only and `quantization_a4w4_noise_0.06.yaml` considering both quantization and noise. Modify the dataset location and weight location in the configuration according to your needs. Then run the following command:

```
python PDT_train.py -C resnet32/training_config/quantization_a4w4_wo_noise.yaml -M resnet32/scripts/resnet32_mapped_ir_PDT.py
```

After running, a `trained_model` directory will be created in the `resnet32` directory, with a corresponding folder for `quantization_a4w4_wo_noise` and its weights and training logs. After training, you can use the weights without noise as the initial weights for noisy training, and then train the weights for the noisy process. The command is as follows:

```
python PDT_train.py -C resnet32/training_config/quantization_a4w4_noise_0.06.yaml -M resnet32/scripts/resnet32_mapped_ir_PDT.py
```

Similarly, after running, a folder for `quantization_a4w4_noise_0.06` will be created in the `resnet32/trained_model` directory with its weights. Note that since the training process under the same configuration occurs multiple times, we use the current time as the file suffix to distinguish different training results. Therefore, the latest trained weight file `resnet32_mapped_ir_PDT_best.pth.tar` in the `quantization_a4w4_noise_0.06` folder is the final training weight. Finally, you can use the trained weights to update the relevant parameters in the IR using the following command:

```
python cimcc.py -i resnet32/ir/resnet32_mapped_ir.yaml --modify_ir -w resnet32/trained_model/quantization_a4w4_noise_0.06/[your_file_name]/resnet32_PDT_best.pth.tar
```

Then, in the `resnet32/ir` directory, the files `resnet32_mapped_ir_with_pdt_weight.yaml` will be created. And the `resnet32/inference_weight/` directory will be created with the file `resnet32_mapped_ir_pdt_weight.pth.tar`.

## 2. Hardware Parameters Optimization

First, we need to generate code for computation parameter optimization based on the `resnet32_mapped_ir_with_pdt_weight.yaml` file. The command is as follows:

```
python cimcc.py -i resnet32/ir/resnet32_mapped_ir_with_pdt_weight.yaml --to_optim
```

After running this, a file named `resnet32_mapped_ir_with_pdt_weight_SIM_OPT.py` will be generated in the `resnet32/scripts/` directory. Then, execute the following command:

```
python optimize.py -i resnet32/ir/resnet32_mapped_ir_with_pdt_weight.yaml -id [training_sample_input_data] -it [training_sample_input_target] -w resnet32/inference_weight/resnet32_mapped_ir_pdt_weight.pth.tar -m resnet32/scripts/resnet32_mapped_ir_with_pdt_weight_SIM_OPT.py -n resnet32_mapped_ir_with_pdt_weight_SIM_OPT
```

Note that `[training_sample_input_target]` here refers to the real-valued outputs of the model, used for calculating loss, not labels. The `[training_sample_input_target]` can be directly inferred using inputs, the model structure, and the weights with PyTorch or ONNX Runtime, without the need to consider any non-ideal factors. The optimized IR file will be generated in the `resnet32/ir/` directory with the name `resnet32_mapped_ir_with_pdt_weight_sil_opt.yaml`.

## 3. Inference Code Generation

Finally, based on the optimized IR, the corresponding hardware executable code can be generated. Here, we use generating simulator code as an example. The command is as follows:

```
python cimcc.py -i resnet32/ir/resnet32_mapped_ir_with_pdt_weight_sil_opt.yaml --to_sim
```

After running this, a file named `resnet32_mapped_ir_with_pdt_weight_sil_opt_SIM.py` will be generated in the `resnet32/scripts/` directory, which is the simulator code. You can then use relevant test data to perform accuracy testing by running the following code:

```
python inference.py -id [test_input_data] -il [test_input_label] -w resnet32/inference_weight/resnet32_mapped_ir_pdt_weight.pth.tar -m resnet32/scripts/resnet32_mapped_ir_with_pdt_weight_sil_opt_SIM.py -n resnet32_mapped_ir_with_pdt_weight_sil_opt_SIM
```

# Citation

If you are using this project in your research, please cite the following paper:

```bibtex
@article{yu2025full,
  title={A full-stack memristor-based computation-in-memory system with software-hardware co-development},
  author={Yu, Ruihua and Wang, Ze and Liu, Qi and Gao, Bin and Hao, Zhenqi and Guo, Tao and Ding, Sanchuan and Zhang, Junyang and Qin, Qi and Wu, Dong and others},
  journal={Nature Communications},
  volume={16},
  number={1},
  pages={2123},
  year={2025},
  publisher={Nature Publishing Group UK London}
}
```

# Discussion

Currently, this project is in its initial version. In the future, we will continue to update the project to meet the needs of various memristor-based compute-in-memory chip architectures and to provide a more convenient development environment.

If you have any problem on using the package or any suggestions, please feel free to create an [issue](https://github.com/Tsinghua-LEMON-Lab/OpenCIMTC/issues) to let us know.
