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

from ikomia import core, dataprocess
from ikomia.utils import pyqtutils, qtconversion
from infer_hf_stable_diffusion.infer_hf_stable_diffusion_process import InferHfStableDiffusionParam
from torch.cuda import is_available
import os
from infer_hf_stable_diffusion.utils import Autocomplete
# PyQt GUI framework
from PyQt5.QtWidgets import *


# --------------------
# - Class which implements widget associated with the process
# - Inherits PyCore.CWorkflowTaskWidget from Ikomia API
# --------------------
class InferHfStableDiffusionWidget(core.CWorkflowTaskWidget):

    def __init__(self, param, parent):
        core.CWorkflowTaskWidget.__init__(self, parent)

        if param is None:
            self.parameters = InferHfStableDiffusionParam()
        else:
            self.parameters = param

        # Create layout : QGridLayout by default
        self.grid_layout = QGridLayout()
        # PyQt -> Qt wrapping
        layout_ptr = qtconversion.PyQtToQt(self.grid_layout)

        # Cuda
        self.check_cuda = pyqtutils.append_check(self.grid_layout,
                                                 "Cuda",
                                                 self.parameters.cuda and is_available())
        self.check_cuda.setEnabled(is_available())

        # Model name
        model_list_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                        "model_list.txt")
        model_list_file = open(model_list_path, "r")

        model_list = model_list_file.read()
        model_list = model_list.split("\n")
        self.combo_model = Autocomplete(model_list, parent=None, i=True, allow_duplicates=False)
        self.label_model = QLabel("Model name")
        self.grid_layout.addWidget(self.combo_model, 0, 1)
        self.grid_layout.addWidget(self.label_model, 0, 0)
        self.combo_model.setCurrentText(self.parameters.model_name)
        model_list_file.close()

        # Prompt
        self.edit_prompt = pyqtutils.append_edit(self.grid_layout, "Prompt", self.parameters.prompt)

        # Number of inference steps
        self.spin_number_of_steps = pyqtutils.append_spin(
                                                    self.grid_layout,
                                                    "Number of steps",
                                                    self.parameters.num_inference_steps,
                                                    min=1, step=1
                                                    )
        # Guidance scale
        self.spin_guidance_scale = pyqtutils.append_double_spin(
                                                        self.grid_layout,
                                                        "Guidance scale",
                                                        self.parameters.guidance_scale,
                                                        min=0, step=0.1, decimals=1
                                                    )

        # Negative prompt
        self.edit_negative_prompt = pyqtutils.append_edit(
                                                    self.grid_layout,
                                                    "Negative prompt",
                                                    self.parameters.negative_prompt
                                                    )


        # Refiner
        # self.check_use_refiner = pyqtutils.append_check(self.grid_layout,
        #                                          "Use refiner",
        #                                          self.parameters.use_refiner)
        

        # Link of some available models
        urlLink = "<a href=\"https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/text2img#diffusers.StableDiffusionPipeline\">"\
                 "Parameters info [Hugging Face] </a>"
        self.qlabelModelLink = QLabel(urlLink)
        self.qlabelModelLink.setOpenExternalLinks(True)
        self.grid_layout.addWidget(self.qlabelModelLink, 10, 1)

        # Set widget layout
        self.set_layout(layout_ptr)

    def on_apply(self):
        # Apply button clicked slot
        self.parameters.update = True
        self.parameters.model_name = self.combo_model.currentText()
        self.parameters.prompt = self.edit_prompt.text()
        self.parameters.num_inference_steps = self.spin_number_of_steps.value()
        self.parameters.guidance_scale = self.spin_guidance_scale.value()
        self.parameters.negative_prompt = self.edit_negative_prompt.text()
        self.parameters.cuda = self.check_cuda.isChecked()
        # self.parameters.use_refiner = self.check_use_refiner.isChecked()

        # Send signal to launch the process
        self.emit_apply(self.parameters)


# --------------------
# - Factory class to build process widget object
# - Inherits PyDataProcess.CWidgetFactory from Ikomia API
# --------------------
class InferHfStableDiffusionWidgetFactory(dataprocess.CWidgetFactory):

    def __init__(self):
        dataprocess.CWidgetFactory.__init__(self)
        # Set the name of the process -> it must be the same as the one declared in the process factory class
        self.name = "infer_hf_stable_diffusion"

    def create(self, param):
        # Create widget object
        return InferHfStableDiffusionWidget(param, None)
