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
import random
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import os


# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class InferHfStableDiffusionParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        # Place default value initialization here
        self.model_name = "stabilityai/stable-diffusion-2-base"
        self.prompt = "Astronaut on Mars during sunset"
        self.cuda = torch.cuda.is_available()
        self.guidance_scale = 7.5
        self.negative_prompt = ""
        self.num_inference_steps = 50
        self.seed = -1
        self.use_refiner = False
        self.update = False

    def set_values(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.model_name = str(param_map["model_name"])
        self.prompt = param_map["prompt"]
        self.cuda = utils.strtobool(param_map["cuda"])
        self.guidance_scale = float(param_map["guidance_scale"])
        self.negative_prompt = param_map["negative_prompt"]
        self.seed = int(param_map["seed"])
        self.num_inference_steps = int(param_map["num_inference_steps"])
        self.use_refiner = utils.strtobool(param_map["use_refiner"])
        self.update = True

    def get_values(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = {}
        param_map["model_name"] = str(self.model_name)
        param_map["prompt"] = str(self.prompt)
        param_map["cuda"] = str(self.cuda)
        param_map["guidance_scale"] = str(self.guidance_scale)
        param_map["negative_prompt"] = str(self.negative_prompt)
        param_map["num_inference_steps"] = str(self.num_inference_steps)
        param_map["seed"] = str(self.seed)
        param_map["use_refiner"] = str(self.use_refiner)
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
        self.seed = None
        self.model_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "weights")


    def get_progress_steps(self):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 1

    def run(self):
        # Core function of your process
        # Call begin_task_run() for initialization
        self.begin_task_run()

        # Get parameters
        param = self.get_param_object()

        # Load pipeline
        if param.update or self.pipe is None:
            self.device = torch.device("cuda") if param.cuda and torch.cuda.is_available() else torch.device("cpu")
            torch_tensor_dtype = torch.float16 if param.cuda and torch.cuda.is_available() else torch.float32

            if param.seed == -1:
                self.seed = random.randint(0, 191965535)
            else:
                self.seed = param.seed

            self.generator = torch.Generator(self.device).manual_seed(param.seed)

            # Load models for the XL version
            if param.model_name == "stabilityai/stable-diffusion-xl-base-1.0":
                try: 
                    self.pipe = DiffusionPipeline.from_pretrained(
                        "stabilityai/stable-diffusion-xl-base-1.0",
                        torch_dtype=torch_tensor_dtype,
                        use_safetensors=True,
                        variant="fp16",
                        cache_dir=self.model_folder,
                        local_files_only=True
                        )
                except Exception as e:
                    print(f"Failed with error: {e}. Trying without the local_files_only parameter...")
                    self.pipe = DiffusionPipeline.from_pretrained(
                        "stabilityai/stable-diffusion-xl-base-1.0",
                        torch_dtype=torch_tensor_dtype,
                        use_safetensors=True,
                        variant="fp16",
                        cache_dir=self.model_folder
                    )    
                if param.use_refiner:
                    refiner = DiffusionPipeline.from_pretrained(
                        "stabilityai/stable-diffusion-xl-refiner-1.0",
                        text_encoder_2=self.pipe.text_encoder_2,
                        vae=self.pipe.vae,
                        torch_dtype=torch.float16,
                        use_safetensors=True,
                        variant="fp16",
                    )

                    refiner = refiner.to(self.device)
                    self.pipe.enable_model_cpu_offload()
                else:
                    self.pipe = self.pipe.to(self.device)

            else:
                try:
                    self.pipe = DiffusionPipeline.from_pretrained(
                                    param.model_name,
                                    torch_dtype=torch_tensor_dtype,
                                    use_safetensors=False,
                                    cache_dir=self.model_folder,
                                    local_files_only=True
                    )
                except Exception as e:
                    print(f"Failed with error: {e}. Trying without the local_files_only parameter...")
                    self.pipe = DiffusionPipeline.from_pretrained(
                                    param.model_name,
                                    torch_dtype=torch_tensor_dtype,
                                    use_safetensors=False,
                                    cache_dir=self.model_folder
                    )                            

                self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
                self.pipe = self.pipe.to(self.device)

                # Enable sliced attention computation
                self.pipe.enable_attention_slicing()

        with torch.no_grad():
            if param.model_name == "stabilityai/stable-diffusion-xl-base-1.0":
                # Inference xl
                result = self.pipe(
                            prompt = param.prompt,
                            output_type = "pil",
                            # output_type = "latent" if param.use_refiner else "pil",
                            generator = self.generator,
                            guidance_scale = param.guidance_scale,
                            negative_prompt = param.negative_prompt,
                            num_inference_steps = param.num_inference_steps,
                            ).images

                if param.use_refiner:
                    result = refiner(
                        prompt = param.prompt,
                        image = result,
                        ).images
            else:
                # Inference
                result = self.pipe(
                                param.prompt,
                                guidance_scale = param.guidance_scale,
                                negative_prompt = param.negative_prompt,
                                generator = self.generator,
                                num_inference_steps = param.num_inference_steps,
                                ).images

        print(f"Prompt:\t{param.prompt}\nSeed:\t{self.seed}")

        # Get and display output 
        image = np.array(result[0])
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
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Diffusion"
        self.info.version = "1.2.1"
        self.info.icon_path = "icons/icon.png"
        self.info.authors = "Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Björn Ommer."
        self.info.article = "High-Resolution Image Synthesis with Latent Diffusion Models"
        self.info.journal = "arXiv"
        self.info.year = 2021
        self.info.license = "MIT License"
        # URL of documentation
        self.info.documentation_link = "https://arxiv.org/pdf/2112.10752.pdf"
        # Code source repository
        self.info.repository = "https://github.com/Ikomia-hub/infer_hf_stable_diffusionn"
        self.info.original_repository = "https://github.com/Stability-AI/stablediffusion"
        # Keywords used for search
        self.info.keywords = "Stable Diffusion, Hugging Face, Stability-AI,text-to-image, Generative"
        self.info.algo_type = core.AlgoType.INFER
        self.info.algo_tasks = "IMAGE_GENERATION"

    def create(self, param=None):
        # Create process object
        return InferHfStableDiffusion(self.info.name, param)
