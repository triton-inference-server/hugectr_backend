from tritonclient.utils import *
import tritonhttpclient  as grpcclient
import tritonhttpclient  as httpclient

import numpy as np

model_name = 'dcn'

with httpclient.InferenceServerClient("localhost:8005") as client:
    input0_data = np.array([[0.005376344086021505,0.0008673026886383349,0.002331002331002331,0.004651162790697674,0.006083972751905593,0.0008793527963418925,0.007590132827324478,0.00591715976331361,0.0420323325635104,0.20000000000000004,0.021428571428571432,0.0,0.00646551724137931]],dtype='float32')#np.random.rand(*shape).astype(np.float32)
    input1_data =np.array([[123,630,1741,169492,439138,549150,549420,559916,561648,562203,595960,617230,785371,951890,954587,961209,1127998,1268021,1272637,1273122,1274952,1284808,1599234,1599246,1661028,1679074,1713689]],dtype='uint32')
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
