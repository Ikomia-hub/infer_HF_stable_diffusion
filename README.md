<div align="center">
  <img src="https://raw.githubusercontent.com/Ikomia-hub/infer_hf_stable_diffusion/main/icons/icon.png" alt="Algorithm icon">
  <h1 align="center">infer_hf_stable_diffusion</h1>
</div>
<br />
<p align="center">
    <a href="https://github.com/Ikomia-hub/infer_hf_stable_diffusion">
        <img alt="Stars" src="https://img.shields.io/github/stars/Ikomia-hub/infer_hf_stable_diffusion">
    </a>
    <a href="https://app.ikomia.ai/hub/">
        <img alt="Website" src="https://img.shields.io/website/http/app.ikomia.ai/en.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/Ikomia-hub/infer_hf_stable_diffusion/blob/main/LICENSE.md">
        <img alt="GitHub" src="https://img.shields.io/github/license/Ikomia-hub/infer_hf_stable_diffusion.svg?color=blue">
    </a>    
    <br>
    <a href="https://discord.com/invite/82Tnw9UGGc">
        <img alt="Discord community" src="https://img.shields.io/badge/Discord-white?style=social&logo=discord">
    </a> 
</p>

Run stable diffusion models from Hugging Face.

![Astronaute xl](https://raw.githubusercontent.com/Ikomia-hubinfer_hf_stable_diffusion/main/icons/output.png)

## :rocket: Use with Ikomia API

#### 1. Install Ikomia API

We strongly recommend using a virtual environment. If you're not sure where to start, we offer a tutorial [here](https://www.ikomia.ai/blog/a-step-by-step-guide-to-creating-virtual-environments-in-python).

```sh
pip install ikomia
```

#### 2. Create your workflow

```python
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name="infer_hf_stable_diffusion", auto_connect=False)

# Run  
wf.run()

# Display the image
display(algo.get_output(0).get_image())
```

## :sunny: Use with Ikomia Studio

Ikomia Studio offers a friendly UI with the same features as the API.

- If you haven't started using Ikomia Studio yet, download and install it from [this page](https://www.ikomia.ai/studio).

- For additional guidance on getting started with Ikomia Studio, check out [this blog post](https://www.ikomia.ai/blog/how-to-get-started-with-ikomia-studio).

## :pencil: Set algorithm parameters

- **model_name** (str) - default 'stabilityai/stable-diffusion-2-base': Name of the stable diffusion model. Other model available:
    - CompVis/stable-diffusion-v1-4
    - runwayml/stable-diffusion-v1-5
    - stabilityai/stable-diffusion-2-base
    - stabilityai/stable-diffusion-2
    - stabilityai/stable-diffusion-2-1-base
    - stabilityai/stable-diffusion-2-1
    - stabilityai/stable-diffusion-xl-base-1.0
- **prompt** (str): Input prompt.
- **negative_prompt** (str, *optional*): The prompt not to guide the image generation. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
- **num_inference_steps** (int) - default '50': Number of denoising steps (minimum: 1; maximum: 500).
- **guidance_scale** (float) - default '7.5': Scale for classifier-free guidance (minimum: 1; maximum: 20).
- **seed** (int) - default '-1': Seed value. '-1' generates a random number between 0 and 191965535.


```python
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name = "infer_hf_stable_diffusion", auto_connect=False)

algo.set_parameters({
    'model_name': 'stabilityai/stable-diffusion-xl-base-1.0',
    'prompt': 'Astronaut on Mars during sunset',
    'guidance_scale': '7.5',
    'negative_prompt': 'low resolution',
    'num_inference_steps': '50',
    'seed': '1981651'
})

# Run directly on your image
wf.run()

# Display the image
display(algo.get_output(0).get_image())
```

## :mag: Explore algorithm outputs

Every algorithm produces specific outputs, yet they can be explored them the same way using the Ikomia API. For a more in-depth understanding of managing algorithm outputs, please refer to the [documentation](https://ikomia-dev.github.io/python-api-documentation/advanced_guide/IO_management.html).

```python
import ikomia
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name="infer_hf_stable_diffusion", auto_connect=False)

# Run 
wf.run()

# Iterate over outputs
for output in algo.get_outputs()
    # Print information
    print(output)
    # Export it to JSON
    output.to_json()
```
