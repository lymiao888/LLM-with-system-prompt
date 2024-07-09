from llama_engine import LLamaEngine
from model.model_metadata import ModelConfig
from sampler.sampling_metadata import SamplingParams
import json


if __name__ == "__main__":
    model_config_llama = ModelConfig("/data/DeepLearning/LLM/models/Llama/Llama-3-8B-Instruct/", "/data/DeepLearning/LLM/models/Llama/Llama-3-8B-Instruct/", True, 1, None)
    LLama = LLamaEngine(model_config_llama)
    sys_prompt = "You can help me find the longest word of the following sentence: "
    requests = [
        ["Hello, my name is Ana. ", 5, 10],
        ["The president of the United States is good. ", 7, 10],
        ["The future of AI is protential", 5, 10]
    ]
    sampling_params = SamplingParams(temperature=1.0, top_p=1.00, max_tokens=512)
    LLama.generate(sys_prompt, requests, sampling_params=sampling_params)

