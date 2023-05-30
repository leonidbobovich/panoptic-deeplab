import onnxruntime as ort


def get_provider():
    # print('providers', ort.get_available_providers())
    provider = 'CPUExecutionProvider'
    for p in ort.get_available_providers():
        if p != provider:
            provider = p
            break
    return provider


def get_input_and_output_sizes(filename, session=None):
    if session is None:
        session = ort.InferenceSession(filename, providers=[get_provider()])
    variables = {}
    inputs = session.get_inputs()
    return_inputs = []
    return_outputs = []
    for i in inputs:
        shape = []
        for s in i.shape:
            if isinstance(s, str):
                if s not in variables.keys():
                    variables[s] = 1
                shape.append(variables[s])
            else:
                shape.append(s)
        return_inputs.append((i.name, shape, i.type))

    outputs = session.get_outputs()
    for o in outputs:
        shape = []
        for s in o.shape:
            if isinstance(s, str):
                if s not in variables.keys():
                    variables[s] = 1
                shape.append(variables[s])
            else:
                shape.append(s)
        return_outputs.append((o.name, shape, o.type))
    return return_inputs, return_outputs, variables
