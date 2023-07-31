/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
/*!
 * \file src/relax/quantize/smooth_quantize/smooth.cc
 * \brief TODO add.
 */
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>
#include <tvm/relax/type.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/dataflow_matcher.h>

#include "../op/tensor/binary.h"
#include "../op/tensor/datatype.h"
#include "../op/tensor/misc.h"
#include "../op/tensor/linear_algebra.h"
#include "../op/tensor/manipulate.h"
#include "../op/tensor/unary.h"
#include "../op/op.h"
#include "utils.h"

#include <tvm/runtime/object.h>

namespace tvm {
namespace relax {

class SmoothQuantConfigNode : public Object {
 public:
  DataType dtype_weight = DataType::Int(8);
  DataType dtype_activation = DataType::Int(8);
  double alpha = 0.5;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("dtype_activation", &dtype_activation);
    v->Visit("dtype_weight", &dtype_weight);
    v->Visit("alpha", &alpha);
  }

  static constexpr const char* _type_key = "relax.quantize.SmoothQuantConfig";
  TVM_DECLARE_BASE_OBJECT_INFO(SmoothQuantConfigNode, Object);
};

/*!
 * \brief Managed reference to dataflow patterns.
 * \sa DFPatternNode
 */
class SmoothQuantConfig : public ObjectRef {
 public:
  static void EnterSmoothQuantConfigScope(const SmoothQuantConfig& config);
  static void ExitSmoothQuantConfigScope();
  static SmoothQuantConfig Current();
  TVM_DEFINE_OBJECT_REF_METHODS(SmoothQuantConfig, ObjectRef, SmoothQuantConfigNode);
};

TVM_REGISTER_NODE_TYPE(SmoothQuantConfigNode);

struct SmoothQuantConfigThreadLocalEntry {
  SmoothQuantConfig default_config;

  /*! \brief The current build config context */
  std::stack<SmoothQuantConfig> config_stack;

  SmoothQuantConfigThreadLocalEntry() : default_config(make_object<SmoothQuantConfigNode>()) {}
};

typedef dmlc::ThreadLocalStore<SmoothQuantConfigThreadLocalEntry> SmoothQuantConfigThreadLocalStore;

void SmoothQuantConfig::EnterSmoothQuantConfigScope(const SmoothQuantConfig& config) {
  SmoothQuantConfigThreadLocalEntry* entry = SmoothQuantConfigThreadLocalStore::Get();
  entry->config_stack.push(config);
}

void SmoothQuantConfig::ExitSmoothQuantConfigScope() {
  SmoothQuantConfigThreadLocalEntry* entry = SmoothQuantConfigThreadLocalStore::Get();
  entry->config_stack.pop();
}

SmoothQuantConfig SmoothQuantConfig::Current() {
  SmoothQuantConfigThreadLocalEntry* entry = SmoothQuantConfigThreadLocalStore::Get();
  if (entry->config_stack.empty()) {
    return entry->default_config;
  }
  return entry->config_stack.top();
}

TVM_REGISTER_GLOBAL("relax._quantize._GetCurrentSmoothQuantConfig").set_body_typed([]() -> SmoothQuantConfig {
  return SmoothQuantConfig::Current();
});

TVM_REGISTER_GLOBAL("relax._quantize._EnterSmoothQuantConfigScope")
    .set_body_typed(SmoothQuantConfig::EnterSmoothQuantConfigScope);

TVM_REGISTER_GLOBAL("relax._quantize._ExitSmoothQuantConfigScope")
    .set_body_typed(SmoothQuantConfig::ExitSmoothQuantConfigScope);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<SmoothQuantConfigNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const SmoothQuantConfigNode*>(node.get());
      p->stream << "dtype_activation(" << op->dtype_activation << "), ";
      p->stream << "dtype_weight(" << op->dtype_weight << "), ";
      p->stream << "alpha(" << op->alpha << ")";
    });

////////////////////////////////////////////////////////////////////////////////////////////////////

class ParamsAndOutputsMutator : public ExprMutator {
 public:
  ParamsAndOutputsMutator() {
    Map<String, ObjectRef> attrs;
    attrs.Set("mode", String("identity"));
    auto lhs_sm = IsOp("relax.annotate.smooth")(Wildcard(), Wildcard()).HasAttr(attrs);
    auto rhs_sm = IsOp("relax.annotate.smooth")(Wildcard(), Wildcard()).HasAttr(attrs);
    permute_ = IsOp("relax.permute_dims")(rhs_sm);
    linear_pat_ = IsOp("relax.matmul")(lhs_sm, permute_);
  }

  Expr Run(Function f) {
    bindings_ = AnalyzeVar2Value(f);
    return VisitExpr(f);
  }

  Expr VisitExpr_(const FunctionNode* op) final {
    Expr new_body = this->VisitExpr(op->body);
    Array<Var> new_params;
    for (auto& param: op->params) {
      if (params_to_remove_.count(param) == 0)
        new_params.push_back(param);
    }
    return Function(new_params, new_body, NullOpt, op->is_pure, op->attrs);
  }

  Expr VisitExpr_(const SeqExprNode* seq_expr) final {
    bool all_blocks_unchanged = true;
    Array<BindingBlock> blocks;
    for (auto block : seq_expr->blocks) {
      BindingBlock new_block = VisitBindingBlock(block);
      blocks.push_back(new_block);
      all_blocks_unchanged &= block.same_as(new_block);
    }

    if (all_blocks_unchanged) {
      return GetRef<Expr>(seq_expr);
    } else {
      if (profile_points_.empty())
        return SeqExpr(blocks, seq_expr->body, seq_expr->span);

      Array<Expr> new_outputs;
      new_outputs.push_back(seq_expr->body);
      new_outputs.push_back(profile_points_.size() == 1 ? profile_points_[0] : Tuple(profile_points_));
      Expr body = Tuple(new_outputs);
      return SeqExpr(blocks, body, seq_expr->span);
    }
  }

  BindingBlock VisitBindingBlock_(const DataflowBlockNode* block) final {
    builder_->BeginDataflowBlock();
    for (Binding binding : block->bindings) {
      this->VisitBinding(binding);
    }
    // Do not emit new outputs if there is no profile points.
    if (!profile_points_.empty()) {
      builder_->EmitOutput(profile_points_.size() == 1 ? profile_points_[0] : Tuple(profile_points_));
    }
    return builder_->EndBlock();
  }

  Expr VisitExpr_(const CallNode* call_node) final {
    Expr new_expr = ExprMutator::VisitExpr_(call_node);
    Call new_call = Downcast<Call>(new_expr);
    if (auto matched_expr = ExtractMatchedExpr(linear_pat_, new_expr, bindings_)) {
      auto permute = Downcast<Call>(matched_expr.value()[permute_]);
      auto mm = Downcast<Call>(matched_expr.value()[linear_pat_]);

      Var a_out = builder_->Emit(absmax(mm->args[0], kSQActivation), "a_out");
      Var w_out = builder_->Emit(absmax(permute->args[0], kSQWeight), "w_out");
      // Remember the place for the new graph outputs.
      profile_points_.push_back(a_out);
      profile_points_.push_back(w_out);
    }
    return new_expr;
  }

  Expr VisitExpr_(const VarNode* op) final {
    std::string name = op->name_hint();
    // substitue var with constant If var name starts with "sq_scale_".
    if (name.rfind("sq_scale_", 0) == 0) {
      Var scale_var = GetRef<Var>(op);
      // Save "smooth multiplier" var to remove it later from function's params.
      params_to_remove_.insert(scale_var);

      DataType dtype = GetDataTypeFromTensor(scale_var);
      auto shape = GetShapeFromTensor(scale_var);
      DLDevice cpu_dev = {DLDeviceType::kDLCPU, 0};
      if (shape.empty()) {
        runtime::NDArray ret_tensor = runtime::NDArray::Empty({}, dtype, cpu_dev);
        return Constant(ret_tensor);
      } else {
        ICHECK(shape.size() == 1);
        int64_t dim = Downcast<IntImm>(shape[0])->value;
        runtime::NDArray ret_tensor = runtime::NDArray::Empty({dim}, dtype, cpu_dev);
        return Constant(ret_tensor);
      }
    }
    return ExprMutator::VisitExpr_(op);
  }

 private:
  DFPattern linear_pat_, permute_;
  Array<Expr> profile_points_;
  std::unordered_set<Var, ObjectHash, ObjectEqual> params_to_remove_;
  Map<Var, Expr> bindings_;
};

class Annotator : public ExprMutator {
 public:
  explicit Annotator(String mode): mode_(mode) {
    permute_ = IsOp("relax.permute_dims")(Wildcard());
    linear_pat_ = IsOp("relax.matmul")(Wildcard(), permute_);

    Map<String, ObjectRef> attrs;
    attrs.Set("mode", String("multiply"));
    auto lhs_sm_ = IsOp("relax.annotate.smooth")(Wildcard(), Wildcard()).HasAttr(attrs);
    auto rhs_sm_ = IsOp("relax.annotate.smooth")(permute_, Wildcard()).HasAttr(attrs);
    smooth_linear_pat_ = IsOp("relax.matmul")(lhs_sm_, rhs_sm_);
  }

  Expr AttachOps(Function f) {
    bindings_ = AnalyzeVar2Value(f);
    return VisitExpr(f);
  }

  Expr VisitExpr_(const FunctionNode* func) final {
    Expr new_body = ExprMutator::VisitExpr(func->body);
    if (new_body.same_as(func->body)) {
      return GetRef<Expr>(func);
    }
    Array<Var> params = func->params;
    params.insert(params.end(), new_params_.begin(), new_params_.end());
    return Function(params, new_body, func->ret_struct_info, func->is_pure, func->attrs, func->span);
  }

  Expr VisitExpr_(const CallNode* call_node) final {
    Expr new_e = ExprMutator::VisitExpr_(call_node);

    if (auto matched_expr = ExtractMatchedExpr(linear_pat_, new_e, bindings_)) {
      auto permute = Downcast<Call>(matched_expr.value()[permute_]);
      auto mm = Downcast<Call>(matched_expr.value()[linear_pat_]);

      Expr act = mm->args[0];
      Expr weights = permute->args[0];

      // Weights of linear op has 2 dimensions. Activation is 2D or 3D tensor.
      if (GetNumDimsFromTensor(weights) != 2) return new_e;
      if (GetNumDimsFromTensor(act) != 2 && GetNumDimsFromTensor(act) != 3) return new_e;

      auto gen_scale_param = [this](Array<PrimExpr> shape, DataType dtype) -> Var {
        int64_t idx = shape.size() - 1; // take last dim in tensor shape.
        //auto var_shape = mode_ == "quantize" ? Array<PrimExpr>() : Array<PrimExpr>({shape[idx]});
        auto var_shape = mode_ == "quantize" ? Array<PrimExpr>({1}) : Array<PrimExpr>({shape[idx]});
        TensorStructInfo sinfo(ShapeExpr(var_shape), dtype);
        String param_name = "sq_scale_" + std::to_string(sm_counter_++);
        Var scale(param_name, sinfo);
        new_params_.push_back(scale);
        return scale;
      };

      DataType a_dtype = GetDataTypeFromTensor(act);
      auto act_shape = GetShapeFromTensor(act);

      DataType w_dtype = GetDataTypeFromTensor(weights);
      auto weights_shape = GetShapeFromTensor(weights);

      Expr sm_lhs = smooth(act, gen_scale_param(act_shape, a_dtype), kSQActivation, "identity");
      Expr sm_rhs = smooth(weights, gen_scale_param(weights_shape, w_dtype), kSQWeight, "identity");

      auto perm_attrs = permute->attrs.as<PermuteDimsAttrs>();
      Expr transpose = permute_dims(sm_rhs, perm_attrs->axes);

      auto attrs = mm->attrs.as<MatmulAttrs>();
      return matmul(sm_lhs, transpose, attrs->out_dtype);
    }
    return new_e;
  }

 private:
  DFPattern linear_pat_, smooth_linear_pat_, permute_;
  const Op& matmul_op_ = Op::Get("relax.matmul");
  Map<Var, Expr> bindings_;
  Array<Var> new_params_;
  size_t sm_counter_ = 0;
  String mode_;
};

// Change mode of execution for relax.annotate.smooth op.
class Legalizer : public ExprMutator {
 public:
  Legalizer(String smooth_op_mode): new_mode(smooth_op_mode) {}

  Expr VisitExpr_(const CallNode* call_node) final {
    Expr new_e = ExprMutator::VisitExpr_(call_node);
    Call new_call = Downcast<Call>(new_e);

    if (new_call->op == profile_op_) {
      auto attrs = new_call->attrs.as<AnnotateSmoothAttrs>();
      if (attrs->mode == "identity")
        return smooth(new_call->args[0], new_call->args[1], attrs->kind, new_mode);
    }
    return new_e;
  }

 private:
  const Op& profile_op_ = Op::Get("relax.annotate.smooth");
  const String new_mode;
};


class Realizer : public ExprMutator {
 public:
  Realizer() {
    Map<String, ObjectRef> attrs;
    attrs.Set("mode", String("quantize"));
    lhs_sm_ = IsOp("relax.annotate.smooth")(Wildcard(), Wildcard()).HasAttr(attrs);
    rhs_sm_ = IsOp("relax.annotate.smooth")(Wildcard(), Wildcard()).HasAttr(attrs);
    permute_ = IsOp("relax.permute_dims")(rhs_sm_);
    matmul_pat_ = IsOp("relax.matmul")(lhs_sm_, permute_);
  }

  Expr Run(Function f) {
    bindings_ = AnalyzeVar2Value(f);
    return VisitExpr(f);
  }

  Expr VisitExpr_(const CallNode* call_node) final {
    Expr new_expr = ExprMutator::VisitExpr_(call_node);
    //Call new_call = Downcast<Call>(new_expr);

    if (auto matched_expr = ExtractMatchedExpr(matmul_pat_, new_expr, bindings_)) {
      const auto* tensor = GetStructInfoAs<TensorStructInfoNode>(matched_expr.value()[matmul_pat_]);
      ICHECK(tensor != nullptr) << "Only support rewriting tensor expr";
      DataType dtype = tensor->dtype;

      auto cfg = SmoothQuantConfig::Current();
      DataType atype = cfg->dtype_activation;
      DataType wtype = cfg->dtype_weight;

      auto lhs_sm = Downcast<Call>(matched_expr.value()[lhs_sm_]);
      auto rhs_sm = Downcast<Call>(matched_expr.value()[rhs_sm_]);
      auto permute = Downcast<Call>(matched_expr.value()[permute_]);

      auto perm_attrs = permute->attrs.as<PermuteDimsAttrs>();
      Expr transpose = permute_dims(MakeQuantize(rhs_sm, wtype), perm_attrs->axes);
      Expr mm = matmul(MakeQuantize(lhs_sm, atype), transpose, DataType::Int(32));
      return MakeDequantize(Downcast<Call>(mm), lhs_sm->args[1], rhs_sm->args[1], dtype);
    }
    return new_expr;
  }

 private:
  Expr MakeQuantize(Call call, DataType out_dtype) {
    DataType dtype = GetDataTypeFromTensor(call->args[0]);
    Expr data = round(divide(call->args[0], call->args[1]));
    // Clip + cast:
    ICHECK(out_dtype.is_int() || out_dtype.is_uint()) << "TypeError: unsupported type" << out_dtype;
    double ubound = static_cast<double>(Downcast<IntImm>(tvm::max_value(out_dtype))->value);
    double lbound = static_cast<double>(Downcast<IntImm>(tvm::min_value(out_dtype))->value);
    PrimValue min_value = PrimValue(FloatImm(dtype, lbound));
    PrimValue max_value = PrimValue(FloatImm(dtype, ubound));
    return astype(clip(data, min_value, max_value), out_dtype);
  }

  Expr MakeDequantize(Call call, Expr scale1, Expr scale2, DataType out_dtype) {
    ICHECK(out_dtype == DataType::Float(16) || out_dtype == DataType::Float(32));
    if (out_dtype == DataType::Float(32)) {
      //float scale1_const = GetScalarFromConstant<float>(scale1);
      //float scale2_const = GetScalarFromConstant<float>(scale2);
      //Constant dq_scale = MakeConstantScalar(scale1_const * scale2_const, out_dtype);
      //return multiply(astype(call, out_dtype), dq_scale);
      return multiply(astype(call, out_dtype), multiply(scale1, scale2));
    } else {
      Expr dq_scale = multiply(astype(scale1, DataType::Float(32)), astype(scale2, DataType::Float(32)));
      Expr out = multiply(astype(call, DataType::Float(32)), dq_scale);
      return astype(out, out_dtype);
    }
  }

  DFPattern lhs_sm_, rhs_sm_, matmul_pat_, permute_;
  Map<Var, Expr> bindings_;
};


IRModule AnnotateFunctions(IRModule m, Array<String> funcs, String mode) {
  std::unordered_set<std::string> func_names_set;
  func_names_set.insert(funcs.begin(), funcs.end());

  IRModuleNode* new_module = m.CopyOnWrite();
  //Map<GlobalVar, BaseFunc> functions = m->functions;
  for (auto& [gv, f] : m->functions) {
    if (const auto* relax_f = f.as<FunctionNode>()) {
      if (func_names_set.count(gv->name_hint)) {
        Function f_after = Downcast<Function>(Annotator(mode).AttachOps(GetRef<Function>(relax_f)));
        new_module->Update(gv, f_after);
      }
    }
  }
  return GetRef<IRModule>(new_module);
}


class StopLiftParamsAnnotator : public ExprMutator {
 public:
  Expr VisitExpr_(const CallNode* call_node) final {
    Expr new_e = ExprMutator::VisitExpr_(call_node);
    Call new_call = Downcast<Call>(new_e);

    if (new_call->op == profile_op_) {
      auto attrs = new_call->attrs.as<AnnotateSmoothAttrs>();
      if (attrs->mode == "multiply" && attrs->kind == kSQWeight) {
        Expr stop = MakeStopLiftParams(new_call->args[0]);
        return smooth(stop, new_call->args[1], attrs->kind, attrs->mode);
      }
    }
    return new_e;
  }

 private:
  const Op& profile_op_ = Op::Get("relax.annotate.smooth");
};


class ReshapeAnnotator : public ExprMutator {
 public:
  Expr VisitExpr_(const CallNode* call_node) final {
    Expr new_e = ExprMutator::VisitExpr_(call_node);
    Call new_call = Downcast<Call>(new_e);

    if (new_call->op == matmul_op_) {
      Expr lhs = new_call->args[0];
      Expr rhs = new_call->args[1];

      // Insert activation reshape for 3Dx2D matmul
      if (GetNumDimsFromTensor(lhs) == 3 && GetNumDimsFromTensor(rhs) == 2) {
        auto lhs_shape = GetShapeFromTensor(lhs);
        if (Downcast<IntImm>(lhs_shape[0])->value != 1)
          return new_e;
        auto out_shape = GetShapeFromTensor(new_e);
        Expr lhs_reshape = reshape(lhs, Array<PrimExpr>({lhs_shape[1], lhs_shape[2]}));
        auto attrs = new_call->attrs.as<MatmulAttrs>();
        Expr mm = matmul(lhs_reshape, rhs, attrs->out_dtype);
        return reshape(mm, out_shape);
      }
    }
    return new_e;
  }

 private:
  const Op& matmul_op_ = Op::Get("relax.matmul");
};


Expr CollectStat(Function f) { return ParamsAndOutputsMutator().Run(f); }

Expr Legalize(Function f, String smooth_op_mode) { return Legalizer(smooth_op_mode).VisitExpr(f); }

Expr Realize(Function f) { return Realizer().Run(f); }

Expr StopLiftParams(Function f) { return StopLiftParamsAnnotator().VisitExpr(f); }

Expr ReshapeMatmul(Function f) { return ReshapeAnnotator().VisitExpr(f); }


namespace transform {

Pass Annotate(Array<String> funcs, String mode) {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =
      [=](IRModule mod, PassContext pc) { return AnnotateFunctions(mod, funcs, mode); };
  return CreateModulePass(pass_func, 0, "Annotate", {});
}

TVM_REGISTER_GLOBAL("relax.transform.Annotate").set_body_typed(Annotate);

Pass CollectStat() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) { return Downcast<Function>(CollectStat(f)); };
  return CreateFunctionPass(pass_func, 0, "CollectStat", {});
}

TVM_REGISTER_GLOBAL("relax.transform.CollectStat").set_body_typed(CollectStat);

Pass SmoothQuantLegalize(String smooth_op_mode) {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) { return Downcast<Function>(Legalize(f, smooth_op_mode)); };
  return CreateFunctionPass(pass_func, 0, "SmoothQuantLegalize", {});
}

TVM_REGISTER_GLOBAL("relax.transform.SmoothQuantLegalize").set_body_typed(SmoothQuantLegalize);

Pass SmoothQuantRealize() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) { return Downcast<Function>(Realize(f)); };
  return CreateFunctionPass(pass_func, 0, "SmoothQuantRealize", {});
}

TVM_REGISTER_GLOBAL("relax.transform.SmoothQuantRealize").set_body_typed(SmoothQuantRealize);

Pass SmoothQuantStopLiftParams() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) { return Downcast<Function>(StopLiftParams(f)); };
  return CreateFunctionPass(pass_func, 0, "SmoothQuantStopLiftParams", {});
}

TVM_REGISTER_GLOBAL("relax.transform.SmoothQuantStopLiftParams").set_body_typed(SmoothQuantStopLiftParams);

Pass SmoothQuantReshapeMatmul() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) { return Downcast<Function>(ReshapeMatmul(f)); };
  return CreateFunctionPass(pass_func, 0, "SmoothQuantReshapeMatmul", {});
}

TVM_REGISTER_GLOBAL("relax.transform.SmoothQuantReshapeMatmul").set_body_typed(SmoothQuantReshapeMatmul);

}  // namespace transform

}  // namespace relax
}  // namespace tvm
