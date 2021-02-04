from tritonclient.utils import *
import tritonhttpclient  as grpcclient
import tritonhttpclient  as httpclient

import numpy as np

model_name = 'deepfm'

with httpclient.InferenceServerClient("localhost:8005") as client:
    input0_data = np.array([[0.0,0.0,0.488888888888889,0.0,0.0,0.037037037037037,0.1111111111111111,0.0604026845637583,0.06,0.2,0.0,0.0,0.0]],dtype='float32')#np.random.rand(*shape).astype(np.float32)
    input1_data =np.array([[58,177,554,811,877,954,1156,1528,1561,1605,1675,1807,2008,2066,2185,2357,2374,2411,2426,2432,2579,2629,2782,2992,3164,3196]],dtype='uint32')
    input2_data = np.array([[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]],dtype='int32')
    inputs = [
        httpclient.InferInput("DES", input0_data.shape,
                              np_to_triton_dtype(input0_data.dtype)),
        httpclient.InferInput("CATCOLUMN", input1_data.shape,
                              np_to_triton_dtype(input1_data.dtype)),
        httpclient.InferInput("ROWINDEX", input2_data.shape,
                              np_to_triton_dtype(input2_data.dtype)),

    ]

    inputs[0].set_data_from_numpy(input0_data)
    inputs[1].set_data_from_numpy(input1_data)
    inputs[2].set_data_from_numpy(input2_data)
    outputs = [
        httpclient.InferRequestedOutput("OUTPUT0")
    ]

    response = client.infer(model_name,
                            inputs,
                            request_id=str(1),
                            outputs=outputs)

    result = response.get_response()

    print(result)
    print(response.as_numpy("OUTPUT0"))
