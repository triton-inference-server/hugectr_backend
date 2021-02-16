from tritonclient.utils import *
import tritonhttpclient  as grpcclient
import tritonhttpclient  as httpclient

import numpy as np

model_name = 'dcn'

with httpclient.InferenceServerClient("localhost:8005") as client:
    input0_data = np.array([[0.0388349514563106,0.1674641148325358,0.0,0.0,0.0,0.0,0.125,0.0268456375838926,0.02,0.0,0.0,0.0,0.0]],dtype='float32')
    input1_data =np.array([[45,112,529,782,836,926,988,1344,1476,1546,1685,1934,1996,2060,2258,2292,2309,2344,2365,2402,2511,2623,2727,3138,3175,3203]],dtype='uint32')
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
