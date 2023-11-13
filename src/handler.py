#!/usr/bin/env python
import inference
import runpod
from runpod.serverless.utils.rp_validator import validate
from typing import Generator, Union
from schema import INPUT_SCHEMA

### Initialize the model to load it into memory
MODEL = inference.Predictor()
MODEL.setup()

def run(job) -> Union[str, Generator[str, None, None]]:
    ### Validate the input
    validated_input = validate(job["input"], INPUT_SCHEMA)
    if "errors" in validated_input:
        yield {"error": validated_input["errors"]}
        return {"error": validated_input["errors"]}
    validated_input = validated_input["validated_input"]

    ### Run the model depending on stream
    prediction = MODEL.predict(
        settings=validated_input
    )

    for chunk in prediction:
        text, input_tokens, output_tokens = chunk
        # Uncomment this to preview the predictions in the console
        # print(text, end="")
        res = {
            "text": text,
            "output_tokens": output_tokens,
        }
        if input_tokens > 0:
            res["input_tokens"] = input_tokens
        
        yield res


runpod.serverless.start({
    "handler": run,
    "return_aggregate_stream": False
})
