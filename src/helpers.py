import tensorflow as tf
import tensorflow_federated as tff

logical_gpus = 8
logical_gpu_memory = 1024


def setup_tff_runtime():
    tff.backends.native.set_sync_local_cpp_execution_context()
    gpu_devices = tf.config.list_physical_devices('GPU')
    if not gpu_devices:
        print(tf.config.list_logical_devices())
        return
    tf.config.set_logical_device_configuration(
        gpu_devices[0], 
        [
            tf.config.LogicalDeviceConfiguration(memory_limit=logical_gpu_memory) 
            for _ in range(logical_gpus)
        ]
    )
    
    print(tf.config.list_logical_devices())

