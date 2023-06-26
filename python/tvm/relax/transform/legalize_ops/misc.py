# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name
"""Default legalization function for other operators."""

import tvm
from tvm import topi
from ...block_builder import BlockBuilder
from ...expr import Call, Constant, Expr, const
from .common import register_legalize


@register_legalize("relax.annotate.smooth")
def _smooth(bb: BlockBuilder, call: Call) -> Expr:
    if call.attrs.mode == "identity":
        return bb.call_te(topi.identity, call.args[0])

    assert call.attrs.mode == "multiply"

    # smooth for activations:
    if call.attrs.kind == 1:
        return bb.call_te(topi.divide, call.args[0], call.args[1])

    # smooth for weights:
    assert call.attrs.kind == 2
    return bb.call_te(topi.multiply, call.args[0], call.args[1])


@register_legalize("relax.annotate.absmax")
def _absmax(bb: BlockBuilder, call: Call) -> Expr:
    def _compute_max(data, axis):
        return topi.squeeze(topi.max(topi.abs(data), axis=axis))
    return bb.call_te(_compute_max, call.args[0], -2)

