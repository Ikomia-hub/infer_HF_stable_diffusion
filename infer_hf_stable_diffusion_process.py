# Copyright (C) 2021 Ikomia SAS
# Contact: https://www.ikomia.com
#
# This file is part of the IkomiaStudio software.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import copy
from ikomia import core, dataprocess, utils
import torch
import numpy as np
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, EulerDiscreteScheduler


# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class InferHfStableDiffusionParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        # Place default value initialization here
        self.model_name = "stabilityai/stable-diffusion-2-base"
        self.prompt = "a photo of an astronaut riding a horse on mars"
        self.cuda = torch.cuda.is_available()
        self.guidance_scale = 7.5
        self.height = 512
        self.width = 512
        self.num_images_per_prompt = 1
        self.negative_prompt = ""
        self.generator = ""
        self.num_inference_steps = 50
        self.update = False

    def set_values(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.model_name = str(param_map["model_name"])
        self.prompt = param_map["prompt"]
        self.cuda = utils.strtobool(param_map["cuda"])
        self.guidance_scale = float(param_map["guidance_scale"])
        self.height = int(param_map["height"])
        self.width = int(param_map["width"])
        self.num_images_per_prompt = int(param_map["num_images_per_prompt"])
        self.negative_prompt = param_map["negative_prompt"]
        self.generator = param_map["generator"]
        self.num_inference_steps = int(param_map["num_inference_steps"])
        self.update = True

    def get_values(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = {}
        param_map["model_name"] = str(self.model_name)
        param_map["prompt"] = str(self.prompt)
        param_map["cuda"] = str(self.cuda)
        param_map["guidance_scale"] = str(self.guidance_scale)
        param_map["height"] = str(self.height)
        param_map["width"] = str(self.width)
        param_map["num_images_per_prompt"] = str(self.num_images_per_prompt)
        param_map["negative_prompt"] = str(self.negative_prompt)
        param_map["generator"] = str(self.generator)
        param_map["num_inference_steps"] = str(self.num_inference_steps)
        return param_map

# --------------------
# - Class which implements the process
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class InferHfStableDiffusion(core.CWorkflowTask):

    def __init__(self, name, param):
        core.CWorkflowTask.__init__(self, name)
        # Add input/output of the process here
        self.add_output(dataprocess.CImageIO())

        # Create parameters class
        if param is None:
            self.set_param_object(InferHfStableDiffusionParam())
        else:
            self.set_param_object(copy.deepcopy(param))

        self.device = torch.device("cpu")
        self.pipe = None
        self.generator = None
        self.height = 512
        self.width = 512

    def get_progress_steps(self):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 1

    def check_img_size(self, num):
        ''' 
        Check if the image size is a multiple of 8 
        and return the closest multiple of 8
        '''
        return num - (num % 8)
    
    def run(self):
        # Core function of your process
        # Call begin_task_run() for initialization
        self.begin_task_run()

        # Get parameters
        param = self.get_param_object()

        # Load pipeline
        if param.update or self.pipe is None:
            self.device = torch.device("cuda") if param.cuda else torch.device("cpu")
            torch_tensor_dtype = torch.float16 if param.cuda else torch.float32

            scheduler = EulerDiscreteScheduler.from_pretrained(
                                                    param.model_name,
                                                    subfolder="scheduler"
                                                    )
            self.pipe = StableDiffusionPipeline.from_pretrained(
                                                    param.model_name,
                                                    torch_dtype=torch_tensor_dtype,
                                                    use_safetensors=False,
                                                    )
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
            self.pipe = self.pipe.to(self.device)

            # Enable sliced attention computation
            self.pipe.enable_attention_slicing()

            if param.generator:
                self.generator = torch.Generator(self.device).manual_seed(int(param.generator))

        # Check if image size
        self.height = self.check_img_size(param.height)
        self.width = self.check_img_size(param.width)

        # Inference
        results = self.pipe(
                        param.prompt,
                        guidance_scale = param.guidance_scale,
                        height = self.height,
                        width = self.width,
                        num_images_per_prompt = param.num_images_per_prompt,
                        negative_prompt = param.negative_prompt,
                        generator = self.generator,
                        num_inference_steps = param.num_inference_steps,
                        ).images

        if len(results) > 1:
            for i, image in enumerate(results):
                self.add_output(dataprocess.CImageIO())
                img = np.array(image)
                output = self.get_output(i)
                output.set_image(img)
        else:
            image = np.array(results[0])
            output_img = self.get_output(0)
            output_img.set_image(image)

        # Step progress bar (Ikomia Studio):
        self.emit_step_progress()

        # Call end_task_run() to finalize process
        self.end_task_run()


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class InferHfStableDiffusionFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "infer_hf_stable_diffusion"
        self.info.short_description = "Stable diffusion models from Hugging Face."
        self.info.description = "This plugin proposes inference for stable diffusion " \
                                "using models from Hugging Face."
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Diffusion"
        self.info.version = "1.0.0"
        self.info.icon_path = "icons/icon.png"
        self.info.authors = "Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Björn Ommer."
        self.info.article = "High-Resolution Image Synthesis with Latent Diffusion Models"
        self.info.journal = "arXiv"
        self.info.year = 2021
        self.info.license = "MIT License"
        # URL of documentation
        self.info.documentation_link = "https://arxiv.org/pdf/2112.10752.pdf"
        # Code source repository
        self.info.repository = "https://github.com/Stability-AI/stablediffusion"
        # Keywords used for search
        self.info.keywords = "stable diffusion,huggingface,Stability-AI,text-to-image,generative"
    def create(self, param=None):
        # Create process object
        return InferHfStableDiffusion(self.info.name, param)
