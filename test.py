import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def test_tensorrt():
    try:
        builder = trt.Builder(TRT_LOGGER)
        config = builder.create_builder_config()
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        print("TensorRT and CUDA are compatible.")
    except Exception as e:
        print("TensorRT and CUDA are not compatible:", e)

test_tensorrt()
