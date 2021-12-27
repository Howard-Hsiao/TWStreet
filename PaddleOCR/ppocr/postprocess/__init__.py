# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy

__all__ = ['build_post_process']

from .db_postprocess import DBPostProcess, DistillationDBPostProcess
from .east_postprocess import EASTPostProcess
from .sast_postprocess import SASTPostProcess
from .rec_postprocess import CTCLabelDecode, AttnLabelDecode, SRNLabelDecode, DistillationCTCLabelDecode, NRTRLabelDecode, \
    TableLabelDecode
from .cls_postprocess import ClsPostProcess
from .pg_postprocess import PGPostProcess

def build_post_process(config, global_config=None):
    support_dict = [
        'DBPostProcess', 'EASTPostProcess', 'SASTPostProcess', 'CTCLabelDecode',
        'AttnLabelDecode', 'ClsPostProcess', 'SRNLabelDecode', 'PGPostProcess',
        'DistillationCTCLabelDecode', 'NRTRLabelDecode', 'TableLabelDecode', 'DistillationDBPostProcess'
    ]

    config = copy.deepcopy(config)
    module_name = config.pop('name')
    if global_config is not None:
        config.update(global_config)
    assert module_name in support_dict, Exception(
        'post process only support {}'.format(support_dict))
    print(module_name)
    module_class = eval(module_name)(**config)
    print(module_class)
    return module_class
