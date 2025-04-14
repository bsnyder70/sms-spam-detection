from transformers import (
    Blip2VisionConfig,
    Blip2QFormerConfig,
    OPTConfig,
    Blip2Config,
    Blip2ForConditionalGeneration,
    AutoProcessor
)

import torch
import requests
from PIL import Image
from IPython.display import display


url = 'https://media.newyorker.com/cartoons/63dc6847be24a6a76d90eb99/master/w_1160,c_limit/230213_a26611_838.jpg'
image = Image.open(requests.get(url, stream=True).raw).convert('RGB')  
display(image.resize((596, 437)))

'''
# Initializing a Blip2Config with Salesforce/blip2-opt-2.7b style configuration
configuration = Blip2Config()

# Initializing a Blip2ForConditionalGeneration (with random weights) from the Salesforce/blip2-opt-2.7b style configuration
model = Blip2ForConditionalGeneration(configuration)

# Accessing the model configuration
configuration = model.config

# We can also initialize a Blip2Config from a Blip2VisionConfig, Blip2QFormerConfig and any PretrainedConfig

# Initializing BLIP-2 vision, BLIP-2 Q-Former and language model configurations
vision_config = Blip2VisionConfig()
qformer_config = Blip2QFormerConfig()
text_config = OPTConfig()

config = Blip2Config.from_text_vision_configs(vision_config, qformer_config, text_config)
'''
processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

print("pre model")
inputs = processor(image, return_tensors="pt").to(device, torch.float16)

generated_ids = model.generate(**inputs, max_new_tokens=20)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

print(generated_text)