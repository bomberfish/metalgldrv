// Compile-time constant evaluation.
#pragma once
#include "glsl/ast.h"

namespace glsl {

using CV = std::shared_ptr<ConstVal>;

int flatCount(const Type& t);
double cvF(const Scalar& s, BT b);
int64_t cvI(const Scalar& s, BT b);
uint64_t cvU(const Scalar& s, BT b);
bool cvB(const Scalar& s, BT b);
Scalar makeScalar(BT b, double f, int64_t i, uint64_t u, bool bo);
Scalar convertScalar(const Scalar& s, BT from, BT to);

CV foldBinary(Op op, const Type& res, const Expr* a, const Expr* b);
CV foldUnary(Op op, const Type& res, const Expr* a);
CV foldCtor(const Type& to, const std::vector<Expr*>& args);
CV foldIndex(const Expr* base, int idx, const Type& res);
CV foldSwizzle(const Expr* base, const uint8_t* swz, int n, const Type& res);
CV foldField(const Expr* base, int field, const Type& res);
CV foldBuiltin(const std::string& name, const Type& ret, const std::vector<Expr*>& args);

}  // namespace glsl
